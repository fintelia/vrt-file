use fnv::FnvHashMap;
use lru::LruCache;
use raster::{Raster, Values};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Deserializer};
use std::collections::VecDeque;
use std::io::Cursor;
use std::mem;
use std::os::unix::prelude::MetadataExt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use tiff::ColorType;
use vec_map::VecMap;

mod query;
mod raster;

const MAX_WEAK_SIZE: usize = 1024;

pub trait Scalar: raster::ScalarInner + Send + Sync + 'static {}
impl Scalar for u8 {}
impl Scalar for i16 {}
impl Scalar for u16 {}

#[derive(Deserialize, Debug)]
#[allow(unused, non_snake_case)]
struct VRTDataset {
    rasterXSize: usize,
    rasterYSize: usize,
    #[serde(deserialize_with = "split_f64s")]
    GeoTransform: Vec<f64>,
    VRTRasterBand: VRTRasterBand,
}

#[derive(Deserialize, Debug)]
#[allow(unused, non_snake_case)]
struct VRTRasterBand {
    ColorInterp: String,
    #[serde(rename = "SimpleSource", default)]
    sources: Vec<SimpleSource>,
}

#[derive(Deserialize, Debug)]
#[allow(unused, non_snake_case)]
struct SimpleSource {
    SourceFilename: SourceFilename,
    SourceBand: u32,
    SourceProperties: SourceProperties,
    SrcRect: Rect,
    DstRect: Rect,
}
#[derive(Deserialize, Debug)]
#[allow(unused, non_snake_case)]
struct SourceFilename {
    relativeToVRT: u8,
    #[serde(rename = "$value")]
    filename: String,
}
#[derive(Deserialize, Debug)]
#[allow(unused, non_snake_case)]
struct SourceProperties {
    RasterXSize: u32,
    RasterYSize: u32,
    BlockXSize: u32,
    BlockYSize: u32,
}
#[derive(Deserialize, Debug, Clone)]
#[allow(unused, non_snake_case)]
struct Rect {
    xOff: f64,
    yOff: f64,
    xSize: f64,
    ySize: f64,
}
impl Rect {
    fn contains(&self, x: f64, y: f64) -> bool {
        x >= self.xOff
            && x <= self.xOff + self.xSize
            && y >= self.yOff
            && y <= self.yOff + self.ySize
    }
}

fn split_f64s<'de, D>(d: D) -> Result<Vec<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    let s = s.replace(",", " ");

    Ok(s.split_ascii_whitespace()
        .map(|v| f64::from_str(v).unwrap())
        .collect())
}

#[derive(Debug)]
enum CacheEntry {
    Pending(Arc<Condvar>, usize),
    Loaded(Arc<Raster>, usize),
}

struct CacheInner {
    pinned: VecMap<CacheEntry>,
    pinned_bytes: u64,

    weak: LruCache<usize, Raster>,
    weak_bytes: u64,

    aux_bytes: u64,
    user_bytes: u64,
}

struct Cache {
    filenames: Vec<PathBuf>,
    inner: Mutex<CacheInner>,
    alloc_cv: Condvar,
    capacity_bytes: u64,
}
impl Cache {
    fn wait_for_capacity(&self, bytes: u64) -> MutexGuard<CacheInner> {
        assert!(bytes <= self.capacity_bytes);

        let inner = self.inner.lock().unwrap();
        let mut inner = self
            .alloc_cv
            .wait_while(inner, |inner| {
                inner.pinned_bytes + inner.aux_bytes + inner.user_bytes + bytes
                    > self.capacity_bytes
            })
            .unwrap();

        assert!(
            inner.pinned_bytes + inner.aux_bytes + inner.user_bytes + bytes <= self.capacity_bytes
        );
        while inner.pinned_bytes + inner.aux_bytes + inner.user_bytes + inner.weak_bytes + bytes
            > self.capacity_bytes
        {
            assert!(inner.weak.len() > 0 || inner.weak_bytes == 0);
            let (_, raster) = inner.weak.pop_lru().unwrap();
            inner.weak_bytes -= raster.num_bytes();
        }

        // println!(
        //     "{:.2}GB {:.2}GB {:.2}GB {:.2}GB {:.2}GB",
        //     ((inner.pinned_bytes) >> 20) as f32 / 1024.0,
        //     ((inner.aux_bytes) >> 20) as f32 / 1024.0,
        //     ((inner.weak_bytes) >> 20) as f32 / 1024.0,
        //     ((inner.user_bytes) >> 20) as f32 / 1024.0,
        //     ((bytes) >> 20) as f32 / 1024.0
        // );

        inner
    }

    fn alloc_aux_bytes(&self, bytes: u64) {
        self.wait_for_capacity(bytes).aux_bytes += bytes;
    }
    fn free_aux_bytes(&self, bytes: u64) {
        self.inner.lock().unwrap().aux_bytes -= bytes;
        self.alloc_cv.notify_all();
    }

    fn decrement_refcount(&self, tile: usize) {
        let mut inner = self.inner.lock().unwrap();
        let refcount = match inner.pinned[tile] {
            CacheEntry::Loaded(_, ref mut refcount) => {
                *refcount -= 1;
                *refcount
            }
            _ => unreachable!(),
        };
        if refcount == 0 {
            match inner.pinned.remove(tile).unwrap() {
                CacheEntry::Loaded(a, 0) => {
                    inner.pinned_bytes -= a.num_bytes();
                    inner.weak_bytes += a.num_bytes();
                    assert!(!inner.weak.contains(&tile));
                    if inner.weak.len() == MAX_WEAK_SIZE {
                        inner.weak_bytes -= inner.weak.pop_lru().unwrap().1.num_bytes();
                    }
                    inner.weak.push(tile, Arc::try_unwrap(a).ok().unwrap());
                    self.alloc_cv.notify_all();
                }
                _ => unreachable!(),
            }
        }
    }

    fn check_mem_usage(inner: &CacheInner) {
        let mut pinned_bytes = 0;
        for raster in inner.pinned.iter() {
            match &raster.1 {
                CacheEntry::Loaded(a, n) => {
                    assert!(*n > 0);
                    pinned_bytes += a.num_bytes();
                }
                _ => {}
            }
        }
        assert_eq!(pinned_bytes, inner.pinned_bytes);

        let mut weak_bytes = 0;
        for raster in inner.weak.iter() {
            weak_bytes += raster.1.num_bytes();
        }
        assert_eq!(weak_bytes, inner.weak_bytes);

        assert!(inner.aux_bytes < (3 << 30));
    }

    fn get<U, F: FnOnce(&Raster) -> U>(&self, tile: usize, f: F) -> U {
        let mut inner = self.inner.lock().unwrap();
        Self::check_mem_usage(&inner);

        let ret = if let Some(entry) = inner.pinned.get_mut(tile) {
            match entry {
                CacheEntry::Loaded(a, ref mut refcount) => {
                    *refcount += 1;
                    let a = Arc::clone(a);
                    drop(inner);
                    f(&*a)
                }
                CacheEntry::Pending(a, refcount) => {
                    *refcount += 1;
                    let a = Arc::clone(a);
                    let inner = a
                        .wait_while(inner, |i| matches!(i.pinned[tile], CacheEntry::Pending(..)))
                        .unwrap();
                    match inner.pinned[tile] {
                        CacheEntry::Loaded(ref a, _) => {
                            let a = Arc::clone(a);
                            drop(inner);
                            f(&*a)
                        }
                        ref a => unreachable!("{:?}", a),
                    }
                }
            }
        } else if let Some((_, raster)) = inner.weak.pop_entry(&tile) {
            let a = Arc::new(raster);
            inner.weak_bytes -= a.num_bytes();
            inner.pinned_bytes += a.num_bytes();
            assert!(inner
                .pinned
                .insert(tile, CacheEntry::Loaded(a.clone(), 1))
                .is_none());
            drop(inner);
            f(&*a)
        } else {
            inner
                .pinned
                .insert(tile, CacheEntry::Pending(Arc::new(Condvar::new()), 1));
            drop(inner);

            let filename = &self.filenames[tile];
            let file_size = std::fs::metadata(filename).unwrap().size();

            self.alloc_aux_bytes(file_size.try_into().unwrap());
            let contents = std::fs::read(filename).unwrap();

            let mut img = tiff::decoder::Decoder::new(Cursor::new(contents)).unwrap();
            let (width, height) = img.dimensions().unwrap();
            let pixel_bytes = match img.colortype().unwrap() {
                ColorType::Gray(bits) => bits as u64 / 8,
                _ => unimplemented!(),
            };

            let raster_bytes = width as u64 * height as u64 * pixel_bytes;
            self.alloc_aux_bytes(raster_bytes);
            let values = match img.read_image().unwrap() {
                tiff::decoder::DecodingResult::U8(mut v) => {
                    v.shrink_to_fit();
                    Values::U8(v)
                }
                tiff::decoder::DecodingResult::U16(mut v) => {
                    v.shrink_to_fit();
                    Values::U16(v)
                }
                tiff::decoder::DecodingResult::I16(mut v) => {
                    v.shrink_to_fit();
                    Values::I16(v)
                }
                tiff::decoder::DecodingResult::F32(mut v) => {
                    v.shrink_to_fit();
                    Values::F32(v)
                }
                _ => unimplemented!(),
            };
            let raster = Raster {
                width: width as usize,
                height: height as usize,
                values,
            };
            assert_eq!(raster.num_bytes(), raster_bytes);

            drop(img);
            self.free_aux_bytes(file_size.try_into().unwrap());

            let a = Arc::new(raster);

            let mut inner = self.inner.lock().unwrap();
            inner.aux_bytes -= raster_bytes;
            match inner.pinned.remove(tile).unwrap() {
                CacheEntry::Pending(cv, refcount) => {
                    inner.pinned_bytes += a.num_bytes();
                    inner
                        .pinned
                        .insert(tile, CacheEntry::Loaded(a.clone(), refcount));
                    cv.notify_all();
                }
                _ => unreachable!(),
            };
            drop(inner);
            f(&*a)
        };
        self.decrement_refcount(tile);
        ret
    }
}

pub struct VrtFile {
    dataset: VRTDataset,

    cache: Cache,
    mapping: query::TileMapper,
}

#[derive(Debug, thiserror::Error)]
pub enum VrtError {
    #[error("Load failed")]
    IoError(#[from] std::io::Error),
    #[error("Parse failed")]
    XmlError(#[from] quick_xml::DeError),
}

impl VrtFile {
    pub fn new(path: &Path) -> Result<Self, VrtError> {
        let directory = path.parent().map(Path::to_owned).unwrap_or(".".into());

        let file_contents = std::fs::read_to_string(path)?;
        let dataset: VRTDataset = quick_xml::de::from_str(&file_contents)?;

        let mut filenames = Vec::new();
        for source in dataset.VRTRasterBand.sources.iter() {
            filenames.push(directory.join(&source.SourceFilename.filename));
        }

        let rects: Vec<_> = dataset
            .VRTRasterBand
            .sources
            .iter()
            .map(|s| s.DstRect.clone())
            .collect();

        Ok(Self {
            dataset,
            mapping: query::TileMapper::new(rects.into_boxed_slice()),
            cache: Cache {
                filenames,
                inner: Mutex::new(CacheInner {
                    pinned: VecMap::new(),
                    pinned_bytes: 0,
                    weak: LruCache::new(MAX_WEAK_SIZE),
                    weak_bytes: 0,
                    aux_bytes: 0,
                    user_bytes: 0,
                }),
                alloc_cv: Condvar::new(),
                capacity_bytes: 8 << 30,
            },
        })
    }

    pub fn alloc_user_bytes(&self, bytes: u64) {
        self.cache.wait_for_capacity(bytes).user_bytes += bytes;
    }

    pub fn free_user_bytes(&self, bytes: u64) {
        self.cache.inner.lock().unwrap().user_bytes -= bytes;
        self.cache.alloc_cv.notify_all();
    }

    /// ```no_build
    /// let x = (longitude - self.dataset.GeoTransform[0]) / self.dataset.GeoTransform[1];
    /// let y = (latitude - self.dataset.GeoTransform[3]) / self.dataset.GeoTransform[5];
    /// ```
    pub fn geotransform(&self) -> &[f64] {
        &self.dataset.GeoTransform
    }

    pub fn batch_lookup<T: Scalar + From<i16>>(&self, xys: &[(f64, f64)], output: &mut [T]) {
        let mut tiles = Vec::new();
        xys.par_iter()
            .map(|&(x, y)| self.mapping.get(x, y))
            .collect_into_vec(&mut tiles);

        let mut unique_tiles = tiles.clone();
        unique_tiles.dedup();
        unique_tiles.sort();
        unique_tiles.dedup();

        self.cache
            .alloc_aux_bytes((output.len() * mem::size_of::<T>()) as u64);

        let mut shuffled: FnvHashMap<u32, VecDeque<_>> = //self.pool.install(|| {
            unique_tiles
                .into_iter()
                .filter(|&t| t != u32::MAX)
                .map(|tile| {
                    let mut output = VecDeque::new();
                    let src = self.dataset.VRTRasterBand.sources[tile as usize]
                        .SrcRect
                        .clone();
                    let dst = self.dataset.VRTRasterBand.sources[tile as usize]
                        .DstRect
                        .clone();
                    self.cache.get(tile as usize, |raster| {
                    for (_, (x, y)) in tiles.iter().zip(xys).filter(|&(&t, _)| t == tile) {
                        let x = ((x - dst.xOff) * src.xSize / dst.xSize + src.xOff - 0.5).round()
                            as usize;
                        let y = ((y - dst.yOff) * src.ySize / dst.ySize + src.yOff - 0.5).round()
                            as usize;
                        let value = T::get_value(&raster.values, y * raster.width + x);
                        output.push_back(value);
                    }
                    });
                    (tile, output)
                })
                .collect()
        /* }) */;

        for (output, tile) in output.iter_mut().zip(tiles) {
            if tile != u32::MAX {
                *output = shuffled.get_mut(&tile).unwrap().pop_front().unwrap();
            }
        }

        drop(shuffled);
        self.cache
            .free_aux_bytes((output.len() * mem::size_of::<T>()) as u64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_FILE: &'static str = r#"<VRTDataset rasterXSize="524436" rasterYSize="625219">
    <SRS dataAxisToSRSAxisMapping="2,1">GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]</SRS>
    <GeoTransform> -1.8000138888888890e+02,  6.8645442488630851e-04,  0.0000000000000000e+00,  8.4000138888888884e+01,  0.0000000000000000e+00, -2.7830240151767151e-04</GeoTransform>
    <VRTRasterBand dataType="Byte" band="1">
      <ColorInterp>Gray</ColorInterp>
      <SimpleSource>
        <SourceFilename relativeToVRT="1">Copernicus_DSM_COG_10_N00_00_E006_00_WBM.tif</SourceFilename>
        <SourceBand>1</SourceBand>
        <SourceProperties RasterXSize="3600" RasterYSize="3600" DataType="Byte" BlockXSize="1024" BlockYSize="1024" />
        <SrcRect xOff="0" yOff="0" xSize="3600" ySize="3600" />
        <DstRect xOff="270959.357616211" yOff="298236.736540449" xSize="1456.76094981196" ySize="3593.21369325842" />
      </SimpleSource>
      <SimpleSource>
        <SourceFilename relativeToVRT="1">Copernicus_DSM_COG_10_N00_00_E009_00_WBM.tif</SourceFilename>
        <SourceBand>1</SourceBand>
        <SourceProperties RasterXSize="3600" RasterYSize="3600" DataType="Byte" BlockXSize="1024" BlockYSize="1024" />
        <SrcRect xOff="0" yOff="0" xSize="3600" ySize="3600" />
        <DstRect xOff="275329.640465647" yOff="298236.736540449" xSize="1456.76094981196" ySize="3593.21369325842" />
      </SimpleSource>
    </VRTRasterBand>
    </VRTDataset>"#;

    #[test]
    fn it_works() {
        let dataset: VRTDataset = quick_xml::de::from_str(SAMPLE_FILE).unwrap();
        println!("{:#?}", dataset);
    }
}

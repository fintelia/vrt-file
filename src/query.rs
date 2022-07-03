use crate::Rect;
use smallvec::SmallVec;

type Entry = SmallVec<[u32; 4]>;

const WIDTH: usize = 2048;
const HEIGHT: usize = 1024;

pub(crate) struct TileMapper {
    minx: f64,
    miny: f64,
    scalex: f64,
    scaley: f64,
    entries: Box<[Entry]>,
    rects: Box<[Rect]>,
}
impl TileMapper {
    pub fn new(rects: Box<[Rect]>) -> Self {
        assert_eq!(std::mem::size_of::<Entry>(), 24);

        let mut minx = std::f64::MAX;
        let mut miny = std::f64::MAX;
        let mut maxx = std::f64::MIN;
        let mut maxy = std::f64::MIN;
        for rect in &*rects {
            minx = minx.min(rect.xOff);
            miny = miny.min(rect.yOff);
            maxx = maxx.max(rect.xOff + rect.xSize);
            maxy = maxy.max(rect.yOff + rect.ySize);
        }

        let scalex = WIDTH as f64 / (maxx - minx);
        let scaley = HEIGHT as f64 / (maxy - miny);

        let mut entries = vec![Entry::new(); WIDTH * HEIGHT];

        for (id, r) in rects.iter().enumerate() {
            let x = (r.xOff - minx) * scalex;
            let y = (r.yOff - miny) * scaley;
            let w = r.xSize * scalex;
            let h = r.ySize * scaley;
            let x0 = x.floor() as usize;
            let y0 = y.floor() as usize;
            let x1 = ((x + w).ceil() as usize).min(WIDTH - 1);
            let y1 = ((y + h).ceil() as usize).min(HEIGHT - 1);
            assert!(x0 < WIDTH);
            assert!(y0 < HEIGHT);
            for y in y0..y1 {
                for x in x0..x1 {
                    entries[y * WIDTH + x].push(id as u32);
                }
            }
        }

        TileMapper {
            minx,
            miny,
            scalex,
            scaley,
            entries: entries.into_boxed_slice(),
            rects,
        }
    }

    // Return the entry for the given tile.
    pub fn get(&self, x: f64, y: f64) -> u32 {
        let x0 = (((x - self.minx) * self.scalex).floor() as usize).min(WIDTH - 1);
        let y0 = (((y - self.miny) * self.scaley).floor() as usize).min(HEIGHT - 1);

        for id in &self.entries[y0 * WIDTH + x0] {
            if self.rects[*id as usize].contains(x, y) {
                return *id;
            }
        }
        u32::MAX
    }
}

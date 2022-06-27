#[derive(Clone, Debug)]
pub enum Values {
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

pub trait ScalarInner {
    fn get_value(values: &Values, i: usize) -> Self;
}
impl ScalarInner for u8 {
    fn get_value(values: &Values, i: usize) -> Self {
        match values {
            Values::U8(vec) => vec[i],
            Values::I16(vec) => vec[i] as u8,
            Values::U16(vec) => vec[i] as u8,
            Values::F32(vec) => vec[i] as u8,
        }
    }
}
impl ScalarInner for i16 {
    fn get_value(values: &Values, i: usize) -> Self {
        match values {
            Values::U8(vec) => vec[i] as i16,
            Values::I16(vec) => vec[i],
            Values::U16(vec) => vec[i] as i16,
            Values::F32(vec) => vec[i] as i16,
        }
    }
}
impl ScalarInner for u16 {
    fn get_value(values: &Values, i: usize) -> Self {
        match values {
            Values::U8(vec) => vec[i] as u16,
            Values::I16(vec) => vec[i] as u16,
            Values::U16(vec) => vec[i],
            Values::F32(vec) => vec[i] as u16,
        }
    }
}
impl ScalarInner for f32 {
    fn get_value(values: &Values, i: usize) -> Self {
        match values {
            Values::U8(vec) => vec[i] as f32,
            Values::I16(vec) => vec[i] as f32,
            Values::U16(vec) => vec[i] as f32,
            Values::F32(vec) => vec[i],
        }
    }
}

#[derive(Clone, Debug)]
pub struct Raster {
    pub width: usize,
    pub height: usize,
    pub values: Values,
}
impl Raster {
    pub fn num_bytes(&self) -> u64 {
        match self.values {
            Values::U8(ref v) => v.len().try_into().unwrap(),
            Values::I16(ref v) => (v.len() * 2).try_into().unwrap(),
            Values::U16(ref v) => (v.len() * 2).try_into().unwrap(),
            Values::F32(ref v) => (v.len() * 4).try_into().unwrap(),
        }
    }
}

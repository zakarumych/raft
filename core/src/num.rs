use core::{cmp::Ordering, fmt};

use crate::error::RuntimeError;

#[derive(Copy, Clone)]
pub enum Number {
    Integer(i64),
    Float(f64),
}

impl Number {
    pub fn neg(self) -> Result<Number, RuntimeError> {
        match self {
            Number::Integer(i) => Ok(Number::Integer(i.wrapping_neg())),
            Number::Float(f) => Ok(Number::Float(-f)),
        }
    }

    pub fn add(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_add(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 + f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) + f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f + (i as f64))),
        }
    }

    pub fn sub(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_sub(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 - f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) - f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f - (i as f64))),
        }
    }

    pub fn mul(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Ok(Number::Integer(i1.wrapping_mul(i2))),
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 * f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) * f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f * (i as f64))),
        }
    }

    pub fn div(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => {
                if i2 == 0 {
                    return Err(RuntimeError::Other("division by zero".into()));
                }
                Ok(Number::Integer(i1 / i2))
            }
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(f1 / f2)),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float((i as f64) / f)),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(f / (i as f64))),
        }
    }

    pub fn pow(self, rhs: Self) -> Result<Number, RuntimeError> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) if i2 >= 0 => {
                Ok(Number::Integer(i1.wrapping_pow(i2 as u32)))
            }
            (Number::Integer(i1), Number::Integer(i2)) => {
                Ok(Number::Float(libm::pow(i1 as f64, i2 as f64)))
            }
            (Number::Float(f1), Number::Float(f2)) => Ok(Number::Float(libm::pow(f1, f2))),
            (Number::Integer(i), Number::Float(f)) => Ok(Number::Float(libm::pow(i as f64, f))),
            (Number::Float(f), Number::Integer(i)) => Ok(Number::Float(libm::pow(f, i as f64))),
        }
    }

    pub fn eq(self, rhs: Self) -> bool {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => i1 == i2,
            (Number::Float(f1), Number::Float(f2)) => f1 == f2,
            (Number::Integer(i), Number::Float(f)) => float_int_eq(f, i),
            (Number::Float(f), Number::Integer(i)) => float_int_eq(f, i),
        }
    }

    pub fn cmp(self, rhs: Self) -> Option<Ordering> {
        match (self, rhs) {
            (Number::Integer(i1), Number::Integer(i2)) => Some(i1.cmp(&i2)),
            (Number::Float(f1), Number::Float(f2)) => f1.partial_cmp(&f2),
            (Number::Integer(i), Number::Float(f)) => float_int_cmp(f, i).map(Ordering::reverse),
            (Number::Float(f), Number::Integer(i)) => float_int_cmp(f, i),
        }
    }

    pub fn as_int(self) -> i64 {
        match self {
            Number::Integer(i) => i,
            Number::Float(f) => f as i64,
        }
    }
}

impl fmt::Debug for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Integer(i) => write!(f, "{i}i"),
            Number::Float(fl) => write!(f, "{fl}f"),
        }
    }
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Integer(i) => write!(f, "{i}"),
            Number::Float(fl) => write!(f, "{fl}"),
        }
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(*other) == Some(Ordering::Equal)
    }
}

impl PartialEq<i64> for Number {
    fn eq(&self, other: &i64) -> bool {
        match self {
            Number::Integer(i) => i == other,
            Number::Float(f) => float_int_eq(*f, *other),
        }
    }
}

impl PartialEq<Number> for i64 {
    fn eq(&self, other: &Number) -> bool {
        match other {
            Number::Integer(i) => self == i,
            Number::Float(f) => float_int_eq(*f, *self),
        }
    }
}

impl PartialEq<f64> for Number {
    fn eq(&self, other: &f64) -> bool {
        match self {
            Number::Integer(i) => float_int_eq(*other, *i),
            Number::Float(f) => *f == *other,
        }
    }
}

impl PartialEq<Number> for f64 {
    fn eq(&self, other: &Number) -> bool {
        match other {
            Number::Integer(i) => float_int_eq(*self, *i),
            Number::Float(f) => *f == *self,
        }
    }
}

impl PartialOrd for Number {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Number::Integer(i1), Number::Integer(i2)) => Some(i1.cmp(i2)),
            (Number::Float(f1), Number::Float(f2)) => f1.partial_cmp(f2),
            (Number::Integer(i), Number::Float(f)) => float_int_cmp(*f, *i).map(Ordering::reverse),
            (Number::Float(f), Number::Integer(i)) => float_int_cmp(*f, *i),
        }
    }
}

impl PartialOrd<i64> for Number {
    fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
        match self {
            Number::Integer(i) => Some(i.cmp(other)),
            Number::Float(f) => float_int_cmp(*f, *other),
        }
    }
}

impl PartialOrd<Number> for i64 {
    fn partial_cmp(&self, other: &Number) -> Option<Ordering> {
        match other {
            Number::Integer(i) => Some(self.cmp(i)),
            Number::Float(f) => float_int_cmp(*f, *self).map(Ordering::reverse),
        }
    }
}

impl PartialOrd<f64> for Number {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        match self {
            Number::Integer(i) => float_int_cmp(*other, *i).map(Ordering::reverse),
            Number::Float(f) => f.partial_cmp(other),
        }
    }
}

impl PartialOrd<Number> for f64 {
    fn partial_cmp(&self, other: &Number) -> Option<Ordering> {
        match other {
            Number::Integer(i) => float_int_cmp(*self, *i),
            Number::Float(f) => f.partial_cmp(self).map(Ordering::reverse),
        }
    }
}

fn float_int_cmp(f: f64, i: i64) -> Option<Ordering> {
    if f.is_nan() {
        return None;
    }

    let upper_bound = (i64::MAX as u64 + 1) as f64;
    let lower_bound = i64::MIN as f64;

    if f >= upper_bound {
        Some(Ordering::Greater)
    } else if f < lower_bound {
        Some(Ordering::Less)
    } else {
        let floor = libm::floor(f);
        let ifloor = floor as i64;
        if ifloor == i {
            if (f - floor) > 0.0 {
                Some(Ordering::Greater)
            } else {
                Some(Ordering::Equal)
            }
        } else if ifloor < i {
            Some(Ordering::Less)
        } else {
            Some(Ordering::Greater)
        }
    }
}

fn float_int_eq(f: f64, i: i64) -> bool {
    if f.is_nan() {
        return false;
    }

    let upper_bound = (i64::MAX as u64 + 1) as f64;
    let lower_bound = i64::MIN as f64;

    if f >= upper_bound {
        false
    } else if f < lower_bound {
        false
    } else {
        let floor = libm::floor(f);
        let ifloor = floor as i64;
        ifloor == i && (f - floor) == 0.0
    }
}

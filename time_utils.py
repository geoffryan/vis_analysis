#!/usr/bin/env python3
from pathlib import Path
import time
from astropy.time import Time, TimeDelta
import astropy.units as units
import astropy.utils.iers
import astropy.utils.data
from mpmath import mp
import numpy as np

# Set precision for arbitrary-precision floats.
mp.dps = 40

# Eq 5.14 of IERS Conventions (2010) Ch. 5. defines these values
# for converting from UT1 to ERA
ERA_A = mp.mpf(779_057_273_264_000_000) / mp.mpf(1e18)
ERA_B = mp.mpf(1_002_737_811_911_354_480) / mp.mpf(1e18)

# Ensure Astropy will download new IERS data when needed
astropy.utils.iers.conf.auto_download = True
# Set the Astropy IERS Refresh time to the minimum allowed (10 days)
astropy.utils.iers.conf.auto_max_age = 10.0


def get_t_inst_ns(seq, frame0_ns, dseq_ns):

    return frame0_ns + seq * dseq_ns


def calc_delta_tai_utc(t):
    r"""
    Calculate the difference TAI - UTC in seconds at time t. This is the number
    of leap seconds at time t.

    Since astropy's internal representation follows the SOFA standard, its
    representation for UTC during a day which contains a leap second is
    non-uniform. So we first compute t in UTC, then break it into a part
    containing whole days (which will have a uniform representation) and the
    remainder, for which we compute the number of seconds manually. Given
    these we can compute the difference in timestamp between TAI and UTC at t.

    Parameters
    ----------
    t : astropy Time object
        The time at which to calculate TAI - UTC

    Returns
    -------
    delta_tai_utc : float
        Value of TAI-UTC in seconds, rounded to nearest 0.1 ns.
    """

    # Get a representation of t in UTC with years, months, days, etc.
    t_utc = t.utc.ymdhms

    # Form a time object for 0h UTC on the beginning of the given day.
    # This will have a numerical representation (in JD) that can be differenced
    # with the TAI represenatation.
    t_utc_d = Time(
        {
            "year": t_utc.year,
            "month": t_utc.month,
            "day": t_utc.day,
            "hour": 0,
            "minute": 0,
            "second": 0,
        },
        scale="utc",
        precision=9,
    )

    # Compute the remaining time from 0h to the given t, in seconds.
    t_utc_s = 3600 * t_utc.hour + 60 * t_utc.minute + t_utc.second

    # Compute the difference (in seconds) for each part of the time
    # representation.  jd1 is typically the larger value, and has whole days.
    dt1 = 86400 * (t.tai.jd1 - t_utc_d.jd1)
    dt2 = 86400 * (t.tai.jd2 - t_utc_d.jd2) - t_utc_s

    # Due to floating point precision we may have accumulated a few picoseconds
    # of error. In the modern era this dt will always be whole number of
    # seconds, so round the total dt to nearest 0.1 ns.
    dt = round(dt1 + dt2, ndigits=10)

    return dt


def calc_astropy_time_from_unix_ns(t_unix_ns):
    r"""
    Constuct an astropy Time object corresponding to a UNIX timestamp in
    nanoseconds.

    Parameters
    ----------
    t_unix_ns : int
        A UNIX timestamp in nanoseconds.

    Returns
    -------
    Astropy Time object
        A Time object representing the given time.
    """

    # Get the nearest (earlier) UNIX time in whole seconds.
    t_unix_s = np.floor(1.0e-9 * np.atleast_1d(t_unix_ns)).astype(int)

    # The remaining nanoseconds from the whole second stamp.
    t_ns = t_unix_ns - 1_000_000_000 * t_unix_s

    # Use the Python time library to convert the UNIX time in seconds to a
    # struct_time containing the UTC calendar date.
    #
    # We cannot do this with Astropy, because on days with Leap Seconds
    # astropy's "unix" time is not a unix time, the Leap Second is smeared
    # throughout the day.

    y = np.empty(t_unix_s.shape, dtype=int)
    m = np.empty(t_unix_s.shape, dtype=int)
    d = np.empty(t_unix_s.shape, dtype=int)
    h = np.empty(t_unix_s.shape, dtype=int)
    mm = np.empty(t_unix_s.shape, dtype=int)
    s = np.empty(t_unix_s.shape, dtype=int)

    for i in range(len(t_unix_s.flat)):
        t_ts = time.gmtime(t_unix_s.flat[i])
        y.flat[i] = t_ts.tm_year
        m.flat[i] = t_ts.tm_mon
        d.flat[i] = t_ts.tm_mday
        h.flat[i] = t_ts.tm_hour
        mm.flat[i] = t_ts.tm_min
        s.flat[i] = t_ts.tm_sec

    if np.ndim(t_unix_ns) == 0:
        y = y[0]
        m = m[0]
        d = d[0]
        h = h[0]
        mm = mm[0]
        s = s[0]
        t_ns = t_ns[0]

    # Unpack the struct_time into an Astropy time object, add back the
    # remaining nanoseconds.
    t = Time(
        {
            "year": y,
            "month": m,
            "day": d,
            "hour": h,
            "minute": mm,
            "second": s + 1.0e-9 * t_ns,
        },
        scale="utc",
        precision=9,
    )

    return t


def calc_astropy_time_from_inst_ns(t_inst_ns, time0_ns):
    r"""
    Constuct an astropy Time object corresponding to an Instrument time in nanoseconds. 
    Parameters
    ----------
    t_inst_ns : int
        An Instrument time in nanoseconds.

    Returns
    -------
    Astropy Time object
        A Time object representing the given time.
    """

    # First calculate t0, a good UNIX time
    t0 = calc_astropy_time_from_unix_ns(time0_ns)

    # Now add the difference from t0 in TAI nanoseconds
    dt = TimeDelta((t_inst_ns - time0_ns) * units.ns, scale="tai")

    return t0 + dt


def calc_unix_ns_from_t(t):
    r"""
    Compute the UNIX timestamp in nanoseconds from given time t.

    Parameters
    ----------
    t : astropy Time object
        The input time

    Returns
    -------
    int
        The corresponding UNIX timestamp in nanoseconds.
    """

    # Get time in UTC broken into calendar date.
    ymdhms = t.utc.ymdhms

    # Get the time at the beginning (0h) of the UTC day.  The astropy UNIX
    # time conversion is not accurate in the middle of a day the day before a
    # leap second.
    t0h = Time(
        {
            "year": ymdhms.year,
            "month": ymdhms.month,
            "day": ymdhms.day,
            "hour": 0,
            "minute": 0,
            "second": 0,
        },
        scale="utc",
        precision=9,
    )

    # Number of nanoseconds elapsed since t0.
    ns_from_0 = round((t - t0h).tai.to_value("ns"))

    # Return the sum of the unix timestamp from the start of the day and the
    # number of nanoseconds elapsed since then.
    return int(t0h.unix) * 1_000_000_000 + ns_from_0


def calc_tai_ns_from_dt(dt):
    r"""
    Compute the number of TAI nanoseconds elapsed over a time interval, rounded
    to the nearest nanosecond. Should be accurate (up to the precision of the
    given dt) so long as dt ~< 200 years.

    Parameters
    ----------
    dt : astropy TimeDelta object
        The input time interval

    Returns
    -------
    int
        The number of nanoseconds (rounded to the nearest nanosecond) for the
        time interval dt
    """

    # Get the time in the TAI scale.
    tai = dt.tai

    # The time is internally represented as the sum of two JD values in float.
    # Convert each of these to nanoseconds, in floating point.
    ns1_f = 86400 * 1e9 * tai.jd1
    ns2_f = 86400 * 1e9 * tai.jd2

    # Round the first component to the nearest integer nanosecond.
    ns1 = round(ns1_f)

    # Compute the floating point remainder nanoseconds from rounding the first
    # part
    dns = ns1_f - ns1

    # Compute the integer nanoseconds from the second part, including the
    # difference from the first rounding.
    ns2 = round(ns2_f + dns)

    # Return the sum.
    return ns1 + ns2


def get_nrot_at_t(t):

    # ERA_A and ERA_B are greater precision than can fit in a float64,
    # so we do this calculation with large precision floats, switching to a
    # normal integer at the end.

    # Kotekan measures UT1 from 2 451 545 JD UT1, the same point
    # as in the definition of ERA
    dt_jd = (t.ut1.jd1 - mp.mpf(2451545.0)) + t.ut1.jd2

    if np.ndim(t) == 0:
        return int(mp.floor(ERA_A + dt_jd * ERA_B))

    nrot = np.empty(t.shape, dtype=int)
    for i in range(len(nrot.flat)):
        nrot.flat[i] = int(mp.floor(ERA_A + dt_jd[i] * ERA_B))

    return nrot

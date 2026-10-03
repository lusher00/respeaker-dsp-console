# respeaker-dsp-console

Sound direction, activity and identification from a ReSpeaker 2-Mics Pi HAT
v2.0, served over HTTP for the bot dashboard and the hailo-tracker page.

One daemon (`server.py`) on the Pi with the HAT:

- **Bearing** of the current sound, -90..+90 deg, + = right of the board's
  front, from the delay between the two mics (GCC-PHAT).
- **Activity**: band level against a tracked noise floor, plus how coherent
  the two mics are, so room noise and hum do not count as a source.
- **Track**: a bearing that ignores single stray frames and fades out when
  the sound stops.
- **Sound ID**: YAMNet (521 AudioSet classes) on LiteRT, run only while there
  is activity.
- **Events**: one record per sound — time, bearing, spread, duration, peak
  level, and the classes heard during it.
- **Console page** at `/` for all of the above, plus spectrum, delay
  response, settings and mic-spacing calibration.

It does not send drive commands. All endpoints are CORS-open JSON, so
anything on the network can read them.

## Hardware

- ReSpeaker 2-Mics Pi HAT **v2.0** (TLV320AIC3104 codec, ALSA card
  `seeed2micvoicec`, overlay `dtoverlay=respeaker-2mic-v2_0`). The v1 HAT
  (WM8960) has different mixer controls; capture and DSP would work, the
  codec settings would not.
- Raspberry Pi Zero 2 W. Direction finding needs only numpy. Sound ID needs
  a **64-bit OS with Python 3.11 or 3.12** (Raspberry Pi OS Bookworm 64-bit,
  or Ubuntu 24.04): those are the only aarch64 wheels `ai-edge-litert`
  publishes. Without them the daemon logs that sound ID is off and runs
  everything else.

### Mic orientation

The bearing is measured from the perpendicular to the line through the two
mics. Mount the HAT with that line across the robot (left-right) and
"front of the board" facing forward. If left and right come out mirrored,
tick **Swap L/R channels**.

## What two mics can and cannot do

- **Front and back look the same.** A source 30 deg right-front and one 150
  deg right-rear produce the same delay; both read +30. The dial draws the
  mirror as a dashed line. Turning toward the reading still ends up facing
  the source: a rear-right source reads right, the robot turns right, the
  source passes 90 deg and from there reads correctly. Only a source dead
  behind (reads 0) produces no turn.
- **Resolution drops toward ±90 deg.** The delay is d·sin(θ)/c, so equal delay
  steps are ~1 deg wide near 0 and ~3 deg wide at 70 deg.
- **One bearing at a time.** Two simultaneous sources produce two peaks in
  the delay response; the track follows the stronger.
- **Reverberation** gives frames with a delay outside the physical limit
  ±d/c. Those are rejected, and the delay response plot shows them.

### Mic spacing

Delay converts to bearing through the mic spacing d. The 58 mm default has
**not been measured** on this board. Either measure centre-to-centre with
calipers and type it into **Mic spacing**, or calibrate: on the page, start
**Calibrate mic spacing**, stand directly off one end of the board (in line
with both mics) and make broadband noise — hiss, rub fabric — for the whole
window. It reports the spacing implied by the 95th-percentile delay and which
side the source was on. Do it from the right: if it says `left`, tick Swap
L/R.

## Install

From the Mac:

```bash
tools/deploy.sh            # rsync to the Pi, run install.sh there
```

Host is the ssh alias `pi0`; override with `RESPEAKER_PI_SSH=...`. VS Code
tasks: deploy, sync, test on Pi, test here, status, logs.

`install.sh` (run on the Pi as the service user, not sudo):

1. apt: `python3-numpy`, `alsa-utils`, `python3-pip`, `curl` if missing
2. `pip install --user ai-edge-litert` if the OS/Python can take it
3. `download_model.sh` → `models/yamnet.tflite` (not in git; rsync leaves it alone)
4. checks `arecord -l` for the card and says what to add to `config.txt` if absent
5. `/etc/default/respeaker-console` from `config/respeaker-console.env.example` (first time only)
6. installs, enables and restarts `respeaker-console.service`

Page: `http://pi0.local:8082/`. Another host's daemon can be viewed with
`?host=name:8082`.

Settings changed from the page (spacing, band, thresholds, PGA gain…)
persist in `/var/lib/respeaker-console/settings.json`. Device, port, and
model path are in `/etc/default/respeaker-console`.

## HTTP

| Method | Path | |
|---|---|---|
| GET | `/state` | bearing, track, levels, current classes. Small; poll at 10 Hz |
| GET | `/events?since=<id>` | sound events newer than `id` |
| GET | `/spectrum` | averaged L/R spectrum (log-spaced) and delay response |
| GET | `/health` | counters, DSP time per hop, capture and classifier status |
| GET/POST | `/config` | runtime settings and limits; POST a partial object |
| POST | `/calibrate` | `{"seconds": 5}` — start a spacing calibration |
| GET | `/calibrate` | progress, then `{"spacing_mm", "frames", "source_side"}` |
| POST | `/track/reset` | clear the track |

`/state`:

```json
{"running": true,
 "level_db": [-52.1, -51.8], "peak_db": [-31.0, -30.2], "band_db": -48.3,
 "frame": {"bearing_deg": 31.2, "coherence": 0.71, "tau_us": -88.4,
           "in_range": true, "active": true, "snr_db": 17.9, "floor_db": -66.2},
 "track": {"tracking": true, "bearing_deg": 30.6, "strength": 9.4, "age_s": 0.02},
 "in_event": true,
 "sound": {"top": [{"label": "Hands", "score": 0.61}, {"label": "Clapping", "score": 0.55}],
           "idle": false, "age_s": 0.12}}
```

- `frame` is the latest analysis frame (every 21 ms at nfft 2048).
  `active` = SNR ≥ `snr_db`, coherence ≥ `coherence_min`, delay physically possible.
- `track.bearing_deg` is null when `tracking` is false. Use `track` to steer
  toward a sound, `frame` to plot it.

An event:

```json
{"id": 12, "time": 1791069161.07, "duration_s": 0.41, "bearing_deg": -24.8,
 "spread_deg": 1.2, "coherence": 0.93, "peak_db": -27.4, "frames": 20,
 "labels": {"Clapping": 0.82, "Hands": 0.64}, "label": "Clapping"}
```

## Tuning

| Setting | Default | Effect |
|---|---|---|
| `band_lo_hz` / `band_hi_hz` | 300 / 4000 | Analysis band for bearing and activity. Raise the low edge to drop motor or fan hum. Above c/(2d) (≈2.96 kHz at 58 mm) one frequency fits more than one delay inside ±d/c; summed over the band the true delay still wins, but sidelobes rise. Ending the band near 3 kHz gives a cleaner peak; going higher keeps more of a clap's energy. |
| `snr_db` | 8 | dB above the noise floor before a frame counts. |
| `coherence_min` | 0.25 | Fraction of the band that must agree on one delay. Diffuse noise sits near 0.05. |
| `floor_rise_db_s` | 3 | How fast the floor climbs to a new steady sound. A constant hum 12 dB up stops counting in ~4 s. |
| `track_half_life_s` | 0.75 | How fast the track forgets. |
| `track_min_strength` | 1.5 | Track evidence needed to report a bearing. A 0.4 s burst reaches ~20. |
| `event_hangover_s` | 0.3 | Quiet time that ends an event. |
| `classify_only_active` | on | Run YAMNet only within 1 s of activity — saves CPU on the Zero. |
| `pga_db` | 16 | Codec mic preamp, 0..59.5 dB. Watch `clipped` in health. |
| `agc` | off | Codec AGC rides each channel separately. The bearing ignores it, the activity gate does not. |

The noise floor starts at the level of the first frame. If the daemon
starts while something is loud, that first sound is absorbed; the floor
drops to the room as soon as it goes quiet.

## Tests

```bash
./tests/run_tests.sh
```

No HAT, amixer or model needed. `tests/fake_capture.py` stands in for
`arecord` (noise bursts from a set bearing), and can drive the real daemon:

```bash
RESPEAKER_CAPTURE_CMD="python3 tests/fake_capture.py --bearing -35" ./server.py
```

## Files

| | |
|---|---|
| `server.py` | daemon: capture, DSP, events, classifier thread, HTTP |
| `doa.py` | GCC-PHAT bearing, noise floor, bearing tracker, decimator |
| `classify.py` | YAMNet wrapper (LiteRT / tflite_runtime / TF) |
| `static/index.html` | console page |
| `models/yamnet_class_map.csv` | YAMNet class names, from tensorflow/models (Apache 2.0) |
| `download_model.sh` | fetches `models/yamnet.tflite` (Google, Apache 2.0) |
| `install.sh`, `uninstall.sh` | service install on the Pi |
| `tools/deploy.sh` | rsync + install from the Mac |
| `systemd/`, `config/` | unit and env template |

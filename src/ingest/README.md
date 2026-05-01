# ingest

Ingest reads a rosbag2 recording and materializes the files downstream
components consume. It does not physically split the bag. Instead it writes virtual
chunk ranges plus decoded per-chunk artifacts under `data/artifacts/raw/<bag_id>/`.

## Outputs

- `bag_meta.json`: source path, duration, topic counts, and bag-level metadata.
- `calibration.json`: frozen per-bag calibration copied from an authored file.
- `chunks/index.parquet`: virtual chunk windows, including overlap ranges.
- `chunks/<chunk_id>/camera_frames.parquet`: one row per decoded camera image.
- `chunks/<chunk_id>/lidar_sweeps.parquet`: one row per LiDAR sweep `.npz`.
- `chunks/<chunk_id>/poses.parquet`: sparse ego poses extracted from `/tf` or `/odom`.
- `chunks/<chunk_id>/frame_index.parquet`: one row per LiDAR sweep and camera pairing.
- `chunks/<chunk_id>/quality.json`: quality metrics and tags.
- `chunks/<chunk_id>/manifest.json`: traceability for inputs, outputs, config, and commit.

The artifact tree is the metadata index for this component. There is no database
service dependency in the current pipeline.

## Source Layout

```text
src/wato_ingest/
|-- cli.py                 # click commands and JSON summaries
|-- config.py              # ingest config schema
|-- runner.py              # end-to-end orchestration for one bag
|-- pipeline.py            # compatibility shim for older imports
|-- inputs/                # bag metadata, chunks, calibration, topic validation
|-- decoders/              # camera, LiDAR, pose extraction, pose interpolation
`-- artifacts/             # frame index, quality report, manifest writer
```

## Common Commands

```bash
python -m wato_ingest inspect-bag --bag /data/bags/example
python -m wato_ingest split --bag /data/bags/example --bag-id example
python -m wato_ingest run --bag /data/bags/example --calibration /config/calibration.json
```

From the repo entrypoint:

```bash
watod -c ingest up
watod run ingest /data/bags/example
watod test ingest
```

# GATKEEPER POPS CLI

POPS loss-prevention analysis for a video: cart tracking, scoring, rule findings.

## Installation

1. Copy your `.mp4` into `sample_videos\`.
2. Double-click `gk_pops.bat` (or `run_demo.bat` for the browser UI).

First run installs everything into `venv_gk-pops-enhanced\` (3-5 min, needs internet). No admin rights, nothing changed outside this folder. Every run after that takes seconds.

## Usage

Interactive (answers questions, shows the command it built):

```
gk_pops.bat
```

Direct command line:

```
gk_pops.bat sample_videos\myclip.mp4 --auto-zones --camera-placement inside_facing_exit --video-path out\ --json-path out\
```
Explicit file paths for each output:

```
gk_pops.bat sample_videos\myclip.mp4 --auto-zones --json-path results\myclip_report.json --video-path results\myclip_annotated.mp4
```

| Flag | Meaning |
|---|---|
| `--camera-placement` | direction of the exit: `outside_facing_entrance`, `inside_facing_exit`, `inside_exit_on_right`, `inside_exit_on_left`, `inside_exit_on_both` |
| `--auto-zones` | use zones already drawn for this video (drawn once in `run_demo.bat`'s Zones step) |
| `--json-path` | where to write the report |
| `--video-path` | also write the annotated video |
| `--dry-run` | validate without running |
| `--force` | overwrite an existing report |
| `--help` | full flag list |



Output: `<name>.json` (scores, events, findings) and `<name>_annotated.mp4` if `--video-path` was set, both in the folder given.

For details beyond this (JSON schema, exit codes, environment internals) see [PROJECT_README.md](PROJECT_README.md).

# approved-photo-symlinks

## Purpose

Maintain `data/approved/` as an extension-correct, symlinked view of the approved photo list returned by `https://crawlr.lawrenz.com/photos.json`.

## Requirements

### Requirement: Sync approved photo symlinks with extensions
The system SHALL iterate through the approved photo list from `https://crawlr.lawrenz.com/photos.json` (paginated by `?page=N`) and, for each approved photo entry, create an extension-correct symlink in `data/approved/` pointing to the corresponding raw file in `data/raw/`.

#### Scenario: Create symlink for an existing raw JPEG
- **WHEN** an approved photo entry has `filename = <name>` and `data/raw/<name>` exists and is detected as a JPEG
- **THEN** the system creates (or updates) a symlink at `data/approved/<name>.jpg` pointing to `../raw/<name>`

#### Scenario: Create symlink for an existing raw PNG
- **WHEN** an approved photo entry has `filename = <name>` and `data/raw/<name>` exists and is detected as a PNG
- **THEN** the system creates (or updates) a symlink at `data/approved/<name>.png` pointing to `../raw/<name>`

#### Scenario: Skip when raw file is missing
- **WHEN** an approved photo entry has `filename = <name>` and `data/raw/<name>` does not exist
- **THEN** the system does not create a symlink in `data/approved/` for that entry

### Requirement: Determine image extension from file bytes
The system SHALL determine the output extension by inspecting the raw file content (magic bytes) rather than by the URL or filename.

#### Scenario: Detect JPEG by magic bytes
- **WHEN** a raw file begins with the JPEG signature bytes
- **THEN** the system treats the file as JPEG and uses extension `jpg`

#### Scenario: Detect PNG by magic bytes
- **WHEN** a raw file begins with the PNG signature bytes
- **THEN** the system treats the file as PNG and uses extension `png`

### Requirement: Idempotent symlink updates
The system SHALL be safe to run multiple times and converge on the same `data/approved/` contents for a given approved list and `data/raw/` state.

#### Scenario: Existing correct symlink is left unchanged
- **WHEN** `data/approved/<name>.<ext>` already exists as a symlink pointing to `../raw/<name>`
- **THEN** the system leaves it unchanged

#### Scenario: Existing incorrect entry is replaced
- **WHEN** `data/approved/<name>.<ext>` exists but is not a symlink to `../raw/<name>`
- **THEN** the system replaces it with a symlink pointing to `../raw/<name>`

### Requirement: Pruning is not performed
The system SHALL NOT remove entries from `data/approved/` based on the approved list.

#### Scenario: Stale symlink remains
- **WHEN** a symlink exists in `data/approved/` whose base filename is not present in the approved list
- **THEN** the system leaves it unchanged

### Requirement: Efficient operation on large directories
The system SHALL process only the approved list entries and SHALL NOT enumerate the entire `data/raw/` directory.

#### Scenario: No directory scan of data/raw
- **WHEN** the system is run
- **THEN** it checks existence of `data/raw/<filename>` by direct path lookup per approved entry, without listing all of `data/raw/`

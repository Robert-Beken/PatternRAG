# PatternRAG Command-Line Options

PatternRAG offers a flexible command-line interface with various options to control document ingestion and processing. These options give you fine-grained control over how documents are processed and added to the vector database.

## Basic Usage

```bash
python ingest.py [options]
```

## Available Command-Line Options

| Option | Description |
|--------|-------------|
| `--force` | Force reprocessing of all documents, even those that were previously processed successfully |
| `--new-vectordb` | Create a new vector database, discarding the existing one (if any) |
| `--reprocess-failed` | Reprocess only documents that previously failed (resulted in 0 chunks) |
| `--verbose` | Enable verbose debugging output |
| `--quiet` | Minimize output messages |
| `--config` | Path to configuration file (class-based version only) |

## Common Usage Scenarios

### Processing New Documents

To process only new documents (default behavior):

```bash
python ingest.py
```

### Reprocessing Failed Documents

If some documents failed to process correctly (resulted in 0 chunks), you can specifically target those for reprocessing:

```bash
python ingest.py --reprocess-failed
```

This is particularly useful when you've fixed issues with document processing and want to retry only the problematic files.

### Complete Reindexing

To completely reprocess all documents:

```bash
python ingest.py --force
```

### Creating a Fresh Vector Database

To create a new vector database (useful if the existing one has issues):

```bash
python ingest.py --new-vectordb
```

You can combine this with other options:

```bash
python ingest.py --force --new-vectordb
```

### Adjusting Verbosity

For minimal output:

```bash
python ingest.py --quiet
```

For detailed debugging information:

```bash
python ingest.py --verbose
```

## Setting Debug Level with Environment Variables

You can also set the debug level using an environment variable:

```bash
# For verbose output
export DEBUG_LEVEL=2
python ingest.py

# For normal output
export DEBUG_LEVEL=1
python ingest.py

# For minimal output
export DEBUG_LEVEL=0
python ingest.py
```

## Troubleshooting Document Processing

If you're encountering issues with document processing:

1. Start with verbose mode to see detailed information:
   ```bash
   python ingest.py --verbose
   ```

2. Check for documents that failed to process:
   ```bash
   python ingest.py --reprocess-failed --verbose
   ```

3. If issues persist, try creating a new vector database:
   ```bash
   python ingest.py --reprocess-failed --new-vectordb
   ```

## Logging Output to a File

To capture detailed logs for troubleshooting:

```bash
python ingest.py --verbose 2>&1 | tee ingest_log.txt
```

This will display the output in the terminal and also save it to `ingest_log.txt` for later analysis.

# build entire project (slow)
build: create-output-dir build-updates

# run all tests (including formatting)
test:
    nixpkgs-fmt --check {{justfile_directory()}}/flake.nix

# download all experiment results locally
sync-output:
    mkdir -p "{{justfile_directory()}}/output/error-counts"
    rclone sync qjit:qjit-data "{{justfile_directory()}}/output"

# build the error-counts container image
build-error-counts: create-output-dir
    #!/usr/bin/env bash
    set -euo pipefail

    # ensure output dir exists
    mkdir -p "{{justfile_directory()}}/output/error-counts"

    # build the image from qjit/error-counts/Dockerfile
    docker build \
      -t error-counts:latest \
      -f "{{justfile_directory()}}/error-counts/Dockerfile" \
      "{{justfile_directory()}}/error-counts"

    # save it as a tarball in the output dir
    docker save error-counts:latest \
      -o "{{justfile_directory()}}/output/error-counts/error-counts.tar"


create-output-dir:
    @mkdir -p "{{justfile_directory()}}/output"

# Compile all .typ files in `/updates` to `/output/updates`
build-updates: create-output-dir
    #!/usr/bin/env bash
    set -euo pipefail
    mkdir -p "{{justfile_directory()}}/output/updates"
    for file in {{justfile_directory()}}/updates/*.typ; do
        if [ -f "$file" ]; then
            echo "Compiling $file..."
            typst compile "$file" "{{justfile_directory()}}/output/updates/$(basename "$file" .typ).pdf"
        fi
    done

watch-update file: build-updates
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Watching {{file}}..."
    typst watch {{file}} "{{justfile_directory()}}/output/updates/$(basename {{file}} .typ).pdf"


fmt:
    nixpkgs-fmt {{justfile_directory()}}/flake.nix

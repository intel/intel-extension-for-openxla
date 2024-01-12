#!/usr/bin/env bash

# TODO: fix it for latest bazel
arg="$1"
shift

if [[ $arg != "@"* ]]; then
    echo "invalid argument"
    exit 1
fi

file_name=${arg#*@}

# Create a temporary directory
tmp_dir="$(mktemp -d -t tmp.XXXXXXXXXX)"
input_object_file=""
{
    read -r ar_flag
    read -r output_file
    while IFS= read -r input_file; do
        if file "$input_file" | grep -q "current ar archive"; then
            ar x "$input_file" --output="$tmp_dir"
        else
            input_object_file="$input_object_file $input_file"
        fi
    done
} < "$file_name"

if [[ $input_object_file != "" ]]; then
    ar "$ar_flag" "$output_file" $input_object_file
else
    ar "$ar_flag" "$output_file" "$tmp_dir"/*
fi

# Remove the temporary directory
rm -rf "$tmp_dir"

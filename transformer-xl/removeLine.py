def remove_every_second_line(file_path, output_file=None):
    # Open the original file and read lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Keep only the lines with even indexes (every other line)
    filtered_lines = [line for i, line in enumerate(lines) if i % 2 == 0]

    # Determine the output file
    output_path = output_file if output_file else file_path

    # Write the filtered lines back to the output file
    with open(output_path, 'w') as file:
        file.writelines(filtered_lines)

    print(f"Every second line removed and saved to {output_path}")

    # Usage
    remove_every_second_line("npzErrorFilesStandford.txt")


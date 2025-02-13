def read_sql_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        sql_query_lines = file.read()

    return "".join(sql_query_lines)

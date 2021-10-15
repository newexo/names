def load_raw(filename):
    with open(filename, encoding="utf-8", errors="ignore") as f:
        return [line.strip() for line in f]


def join_raw(lines, ljust=100):
    return "\n".join([name.ljust(ljust) for name in lines])

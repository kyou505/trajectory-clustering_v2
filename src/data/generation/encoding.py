"""沿用原生成器的 Geohash 与 token 编码规则，保持词表顺序。"""

SPECIAL_TOKENS = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[MASK]": 3,
}


GEOHASH_ALPHABET = "0123456789bcdefghjkmnpqrstuvwxyz"


def encode_geohash(longitude, latitude, precision=7):
    """无额外依赖的标准 Geohash 编码。"""
    longitude_interval = [-180.0, 180.0]
    latitude_interval = [-90.0, 90.0]
    bits = (16, 8, 4, 2, 1)
    bit_index = 0
    character = 0
    even_bit = True
    result = []

    while len(result) < precision:
        interval = longitude_interval if even_bit else latitude_interval
        value = longitude if even_bit else latitude
        midpoint = 0.5 * (interval[0] + interval[1])
        if value >= midpoint:
            character |= bits[bit_index]
            interval[0] = midpoint
        else:
            interval[1] = midpoint
        even_bit = not even_bit
        if bit_index < 4:
            bit_index += 1
        else:
            result.append(GEOHASH_ALPHABET[character])
            bit_index = 0
            character = 0
    return "".join(result)


def encode_and_pad_rows(rows, max_sequence_length):
    geohashes = sorted({token for row in rows for token in row["trajHash"]})
    raw_location_id = {token: index for index, token in enumerate(geohashes)}
    pad_count_error = 0
    for row in rows:
        if row["trajLen"] > max_sequence_length:
            pad_count_error += 1
            continue
        pad_count = max_sequence_length - row["trajLen"]
        locations = [str(raw_location_id[token]) for token in row["trajHash"]]
        times = [str(value) for value in row.pop("time_values")]
        row["trajectory"] = " ".join(locations + ["[PAD]"] * pad_count)
        row["time"] = " ".join(times + ["[PAD]"] * pad_count)
    if pad_count_error:
        raise ValueError(
            f"{pad_count_error} generated trajectories exceed max-sequence-length"
        )
    location_vocab = SPECIAL_TOKENS.copy()
    for raw_id in range(len(geohashes)):
        location_vocab[str(raw_id)] = len(location_vocab)
    return location_vocab

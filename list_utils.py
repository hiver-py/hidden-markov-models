def text_to_list(text, num_axes, data_type=float):
    elements = text.split()
    axis_lengths = [int(element) for element in elements[:num_axes]]
    values = [data_type(element) for element in elements[num_axes:]]
    if num_axes == 1:
        assert len(values) == axis_lengths[0]
        return values
    if num_axes == 2:
        assert len(values) == axis_lengths[0] * axis_lengths[1]
        return [
            [
                values[row * axis_lengths[1] + column]
                for column in range(axis_lengths[1])
            ]
            for row in range(axis_lengths[0])
        ]
    raise ValueError(f'text_to_list cannot handle {num_axes}-dimensional lists')


def list_to_text(list, num_axes):
    if num_axes == 1:
        return f'{len(list)} ' + ' '.join(str(value) for value in list)
    if num_axes == 2:
        return f'{len(list)} {len(list[0])} ' + ' '.join(str(value) for sub in list for value in sub)
    raise ValueError(f'list_to_text cannot handle {num_axes}-dimensional lists')


def vector_scalar_mult(vector, scalar):
    return [v * scalar for v in vector]


def values_close(a, b, num_axes, tolerance=1e-3):
    if num_axes == 0:
        return abs(a - b) < tolerance
    assert len(a) == len(b)
    return all(
        values_close(a_part, b_part, num_axes=num_axes-1, tolerance=tolerance)
        for a_part, b_part in zip(a, b)
    )
    raise ValueError(f'values_close cannot handle {num_axes}-dimensional lists')
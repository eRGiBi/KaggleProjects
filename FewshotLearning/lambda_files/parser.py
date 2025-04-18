def parse_input(inp) -> str:
    """Formats a dictionary into a string to be given to a prompt."""

    spec_text = ""
    for key, value in inp["Specifications"].items():
        spec_text += str(key) + ": " + str(value) + ", "
    spec_text = spec_text[:-2] if spec_text else ""
    
    return f'Name: {inp["Name"]}, Brand: {inp["Brand"]}, \
        Retail Price: {inp["Retail Price"]}, Discounted Price: {inp["Discounted Price"]}, \
        Specifications: {spec_text}.'

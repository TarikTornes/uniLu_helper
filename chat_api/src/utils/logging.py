import datetime, textwrap
from .load_resources import load_configs


def log(category, message, query=None, sep_len= 95, wrap_long_lines=True):
    cfg = load_configs()
    log_file = cfg["general"]["logs"]
    chunk_file = cfg["general"]["chunks"]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if category == "QUERY_RESULTS":
        with open(chunk_file, 'a') as file:
            file.write(message + "\n\n\n")
            file.write("\n\n\n")

    elif category == "QUERY":
        separator = '#' * sep_len
        separator = separator + "\n" + separator
        block = [separator, 'o', f'o {timestamp}', 'o']
        for line in query.splitlines() or ['']:
            if wrap_long_lines:
                wrapped = textwrap.wrap(line,
                                        width=sep_len - 2,
                                        replace_whitespace=False,
                                        drop_whitespace=False,
                                        break_long_words=True)
                if not wrapped:
                    block.append("o")
                else:
                    for segment in wrapped:
                        block.append(f'o {segment}')
            else:
                block.append(f'o {line}')
        
        block.extend(['o', separator])

        with open(chunk_file, 'a') as file:
            file.write('\n'.join(block) + '\n')

    else:
        log_entry = f"[{timestamp}] {category}: {message}\n"
        with open(log_file, 'a') as file:
            file.write(log_entry)

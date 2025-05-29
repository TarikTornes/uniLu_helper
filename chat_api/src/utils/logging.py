import datetime, textwrap
from .load_resources import load_configs
from .format_docs import format_documents


def log(category, message):
    cfg = load_configs()
    log_file = cfg["general"]["logs"]
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {category}: {message}\n"
    with open(log_file, 'a') as file:
        file.write(log_entry)


def log_query(context, query=None, sep_len= 95, wrap_long_lines=True, overwrite=False):
    cfg = load_configs()
    if overwrite:
        chunk_file = cfg["general"]["chunks_last"]
        write_mode = 'w'
    else:
        chunk_file = cfg["general"]["chunks_all"]
        write_mode = 'a'
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


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
        
    block.extend(['o', separator, '\n', format_documents(context)])

    with open(chunk_file, write_mode) as file:
        file.write("\n".join(block) + "\n")









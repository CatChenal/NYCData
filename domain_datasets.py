# ... DOMAIN datasets ....................................................
def get_domain_datasets(cli, local_file=None, replace=False):
    """
    Download results from cli.datasets() call and save into local_file
    if file does not exists or replace=True.
    Inputs:
    cli (object): sodapy.socrata client
    local_file (None or Path): where GET results was/is to be stored; 
                if not set (default): no file saving.
                => overrides replace.
    replace (bool): if local_file is provided, the local copy
                    is (deleted and) replaced with GET results.
    """
    if local_file is None:
        found = Path(local_file).exists()

        if found:
            if replace:
                # query the client for fresh download & save:
                dom_datasets = cli.datasets()
                save_file(local_file, dom_datasets, replace=replace)
                
            return json.load(local_file)
        else:
            if not replace:
                print(f'Local file ({local_file}) not found, but replace==False.\nTry replace=True instead?')
    else:
        # in memory only:
        return cli.datasets()
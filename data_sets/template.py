# One of the following is required (DOWNLOAD)
HF_REPO=""
def DOWNLOAD(download_dir):
    pass

# Always Required
def CONVERT(download_dir):
    """
    Yields batches converting the data into a standard format:
        List [
            Dict {
                'data': str,
                'uuid': str (optional),
                'data_source': str (optional, default=NAME),
                'group_id': str (optional, default='DATE-TIME'),
                'data_type': str (optional, default='language'),
                'files': List[str] (optional, default=[])
            }
        ]
    """
    pass


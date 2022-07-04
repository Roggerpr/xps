def test_write_processed_xp():
    import os
    ftest = '/Users/pabloherrero/sabat/xps_spectra/ITO_ox/0_2020_01_15_ITO_untreated/20200115_ITO_untreated.xy'
    xptest = xps_data_import(path = ftest, name = 'exsitu', label='Ex-situ cleaning')   # Import raw file
    filepath, filename = os.path.split(ftest)
    newpath = filepath + '/proc/'
    try:
        os.mkdir(newpath)
    except FileExistsError:
        pass
    write_processed_xp(newpath + filename, xptest)   # Write identical xp (unprocessed)
    xpproc = read_processed_xp(newpath + filename)   # Read back
    assert (xpproc.dfx.any() == xptest.dfx.any()).all()   # Test whether both df are equal (up to decimal rounding)
    assert 'dfx' in xpproc.__dict__.keys()
    assert 'name' in xpproc.__dict__.keys()
    assert 'path' in xpproc.__dict__.keys()
    assert 'date' in xpproc.__dict__.keys()

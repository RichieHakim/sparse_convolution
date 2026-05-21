import pathlib


def test_readme_documents_precomputed_matrix_size_guidance():
    readme = pathlib.Path('README.md').read_text()

    assert 'Matrix size guidance' in readme
    assert 'precomputed' in readme
    assert '1e8' in readme
    assert "method='lazy'" in readme
    assert "method='gather_scatter'" in readme

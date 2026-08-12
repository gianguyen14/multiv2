def test_backend_imports():
    import backend
    import backend.app
    assert backend is not None
    assert backend.app is not None

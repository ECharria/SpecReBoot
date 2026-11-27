def pytest_addoption(parser):
    """Add CLI flag to overwrite/update the golden reference file."""
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Overwrite the existing golden_behavior.json snapshot with current execution results.",
    )
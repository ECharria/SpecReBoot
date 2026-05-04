import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)  # this makes sure relative imports don't break when our script is called from a directory other than "SpecreBoot\specreboot"

import yaml

from pathlib import Path
from argparse import Namespace, ArgumentParser
from tqdm import tqdm


VERBOSE = True

COSINE_ALIASES          = ["cos", "cosine"]
MODIFIED_COSINE_ALIASES = ["modcos", "modified_cosine", "modified cosine"]
MS2DEEPSCORE_ALIASES    = ["ms2ds", "ms2dp", "ms2deepscore"]
SPEC2VEC_ALIASES        = ["s2v", "spec2vec"]

ALL_ALIASES = COSINE_ALIASES + MODIFIED_COSINE_ALIASES + SPEC2VEC_ALIASES + MS2DEEPSCORE_ALIASES


def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--config-file",                type=Path,  help="file location of default arguments, must be a .yaml file", default="configs\library_matching.yaml")
    parser.add_argument("--library",                    type=Path,  help="mgf file with all library spectra")
    parser.add_argument("--query",                      type=Path,  help="mgf file with all query spectra that are checked for mathces in the library")
    parser.add_argument("--library-cleaned",            type=Path,  help="pre-cleaned mgf file with library spectra")
    parser.add_argument("--query-cleaned",              type=Path,  help="pre-cleaned mgf file with query spectra")
    parser.add_argument("--precursor-tolerance",        type=float, help="Da window for precursor m/z filtering")
    parser.add_argument("--analog-search",              type=bool,  help="flag to enable analog fill-up when exact matches are below top-N")
    parser.add_argument("--top-n",                      type=Path,  help="number of candidates to include in bootstrap phase")
    parser.add_argument("--B",                          type=int,   help="number of bootstrap replicates")
    parser.add_argument("--similarity-type",            type=str,   help="type of similarity metric used to find matches in the library", choices=ALL_ALIASES)
    parser.add_argument("--score-threshold",            type=float, help="minimum bootstrap score for match support calculation")
    parser.add_argument("--outdir",                     type=Path,  help="output directory for CSV results")
    parser.add_argument("--flash-tolerance",            type=float, help="flash tolerance used to compute cosine and modified cosine similarities")
    parser.add_argument("--binning-decimals",           type=int,   help="precision of binning used for masking of spectral peaks")
    parser.add_argument("--seed",                       type=int,   help="seed used to generate random bootstraps")
    parser.add_argument("--ms2deepscore_model_path",    type=Path,  help="model path of ms2deepscore, only required when using ms2deepscore")
    parser.add_argument("--spec2vec_model_path",        type=Path,  help="model path of spec2vec, only required when using spec2vec")
    return parser


def collect_args(parser: ArgumentParser) -> Namespace:
    cli_params = parser.parse_args()
    cli_params = Namespace( **{k: v for k, v in vars(cli_params).items() if v is not None} )

    # load the default parameters
    with open(str(cli_params.config_file), "r") as f:
        default_params = yaml.safe_load(f)
    default_params = Namespace(**{k: v for k, v in default_params.items() if v is not None} )

    # user specified cli params override default params
    params = Namespace( **( vars(default_params) | vars(cli_params)))

    params.similarity_type = params.similarity_type.lower()

    return params


def check_args(parser, params) -> None:
     # check if we have all required arguments
    optional_args      = ["help", "library", "library_cleaned", "query", "query_cleaned", "ms2deepscore_model_path", "spec2vec_model_path"]
    required_arg_names = [action.dest for action in parser._actions if action.dest not in optional_args]

    missing_required_args = [arg_name for arg_name in required_arg_names if not hasattr(params, arg_name)]
    if missing_required_args:
        raise ValueError(f"missing required argument(s): {missing_required_args}")
    
    # query and library input is flexible, but at least one version must exist to work with
    assert hasattr(params, "library") or hasattr(params, "library_cleaned"), 'user must specify either "library", "library_cleaned" arguments or both'
    assert hasattr(params, "query") or hasattr(params, "query_cleaned")    , 'user must specify either "query", "query_cleaned" arguments or both'

    # add default cleaned names if not specified
    if not hasattr(params, "library_cleaned"):
        params.library = Path(params.library)
        params.library_cleaned = params.library.parent / f"{params.library.stem}_cleaned{params.library.suffix}"

    if not hasattr(params, "query_cleaned"):
        params.query = Path(params.query)
        params.query_cleaned = Path() / params.query.parent / f"{params.query.stem}_cleaned{params.query.suffix}"

    # check if option for similarity metric is right
    assert params.similarity_type in ALL_ALIASES, f"unknown option {params.similarity_type}, available options are {ALL_ALIASES}"

    if params.similarity_type in SPEC2VEC_ALIASES:
        assert hasattr(params, "spec2vec_model_path")   , f"spec2vec_model_path must be specified when library matching with {params.similarity_type}"
        assert Path(params.spec2vec_model_path.exists()), f"spec2vec_model_path must exist when library matching with {params.similarity_type}"

    if params.similarity_type in MS2DEEPSCORE_ALIASES:
        assert hasattr(params, "ms2deepscore_model_path")   , f"ms2deepscore_model_path must be specified when library matching with {params.similarity_type}"
        assert Path(params.ms2deepscore_model_path.exists()), f"ms2deepscore_model_path must exist when library matching with {params.similarity_type}"

    # print args for user verification
    log("> running library matching with params:")
    for k, v in sorted(vars(params).items()):
        log(f"    --{k:<35} {v}")
    log()


# immediately collect the arguments before heavy imports to optimize cli interface response time, this makes it break early if something is specified wrongly
if __name__ == "__main__":
    main_parser = build_parser()
    main_params = collect_args(main_parser)
    check_args(main_parser, main_params)
    log("> heavy imports...")


# heavy imports, this might take up to tens of seconds
import gensim

from matchms.importing import load_from_mgf
from matchms.similarity.FlashSimilarity import FlashSimilarity
from specreboot.preprocessing.filtering import general_cleaning
from specreboot.library.library_matching import confidence_aware_match, collect_results, get_spectrum_id, summarize_top_annotation
from ms2deepscore.models import load_model
from ms2deepscore import MS2DeepScore
from spec2vec import Spec2Vec


def main(params: Namespace) -> None:
    log("> running main process...")
    query_spectra   = _clean_spectra(getattr(params, "query", None), getattr(params, "query_cleaned", None))
    library_spectra = _clean_spectra(getattr(params, "library", None), getattr(params, "library_cleaned", None))

    similarity_metric = _get_similarity(params.similarity_type, params.flash_tolerance, getattr(params, "ms2deepscore_model_path", None), getattr(params, "spec2vec_model_path", None))

    results, skipped = [], []

    for query in tqdm(query_spectra, disable=(not VERBOSE)):
        try:
            result = confidence_aware_match(
                query_spectrum              = query,
                library_spectra             = library_spectra,
                similarity_metric           = similarity_metric,
                B                           = params.B,
                top_n                       = params.top_n,
                score_threshold             = params.score_threshold,
                decimals                    = params.binning_decimals,
                seed                        = params.seed,
                precursor_mz_tolerance_da   = params.precursor_tolerance,
                analog_search               = params.analog_search,
            )
            results.append(result)
            log(f"{result.query_id}: {summarize_top_annotation(result)}")
        except ValueError as e:
            query_id = get_spectrum_id(query)
            log(f"Skipping {query_id}: {e}")
            skipped.append(query_id)

    if not results:
        log("> WARNING: no results generated")
        return
    
    top_hits, all_stats = collect_results(results, outdir=params.outdir, prefix="test_cosine")

    log(f"\n> Saved results to {params.outdir}")
    log(top_hits)


def _get_similarity(method_name: str, flash_tolerance: float, ms2deepscore_model_path: Path | None = None, spec2vec_model_path: Path | None = None):
    if method_name in COSINE_ALIASES:
        return FlashSimilarity(score_type="cosine", matching_mode="fragment", tolerance=flash_tolerance)
        
    if method_name in MODIFIED_COSINE_ALIASES:
        return FlashSimilarity(score_type="cosine", matching_mode="hybrid", tolerance=flash_tolerance)
        
    if method_name in MS2DEEPSCORE_ALIASES:
        ms2dp_model = load_model(str(ms2deepscore_model_path))
        return MS2DeepScore(ms2dp_model, progress_bar=False)
        
    if method_name in SPEC2VEC_ALIASES:
        w2v = gensim.models.Word2Vec.load(str(spec2vec_model_path))
        return Spec2Vec(model=w2v, intensity_weighting_power=0.5, allowed_missing_percentage=5.0, progress_bar=False)
        
    raise ValueError(f"unknown option {method_name.lower()}: available options: \n\t{COSINE_ALIASES} \n\t{MODIFIED_COSINE_ALIASES} \n\t{MS2DEEPSCORE_ALIASES} \n\t{SPEC2VEC_ALIASES}")


def _clean_spectra(input_path: Path | str, output_path: Path | str):
    if Path(output_path).exists():
 
        log(f"> loading spectra from {output_path}...")
        spectra = list(tqdm(load_from_mgf(output_path), disable=(not VERBOSE)))
        log(f"> Loaded {len(spectra)} pre-cleaned spectra from {output_path}")

        return spectra
    

    log(f"> cleaned file {output_path} does not exist, creating cleaned file from {input_path}...")
    spectra = list(tqdm(load_from_mgf(input_path), disable=(not VERBOSE), desc=f"loading spectra from {input_path}"))
    cleaned_spectra, _report = general_cleaning(spectra, file_name=str(output_path))

    log(f"> Cleaned and saved {len(spectra)} for {input_path} spectra to {output_path}")
    return cleaned_spectra


if __name__ == "__main__":
    main(main_params)

import os
import io
import base64
import json
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import itertools
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Circle, Rectangle
import argparse 
import warnings
from typing import Optional, List, Dict, Tuple, Any
import re
import html
import pathlib

# Custom model classes and tangermeme imports
try:
    import tfbs.nn.models as models
    from captum.attr import DeepLiftShap, InputXGradient
    
    # Import necessary functions from tangermeme
    from tangermeme.utils import random_one_hot, one_hot_encode, pwm_consensus
    from tangermeme.predict import predict as tpred
    from tangermeme.plot import plot_logo
    from tangermeme.ersatz import substitute
    from tangermeme.marginalize import marginalize
    from tangermeme.io import read_meme, extract_loci
    from tangermeme.seqlet import recursive_seqlets
    from tangermeme.annotate import annotate_seqlets, count_annotations
    from tangermeme.deep_lift_shap import deep_lift_shap as tangermeme_deep_lift_shap 
    from tangermeme.deep_lift_shap import _captum_deep_lift_shap, hypothetical_attributions
    from tangermeme.variant_effect import substitution_effect, deletion_effect, insertion_effect
    try:
        # tangermeme >= 1.0 moved this out of tangermeme.ism.
        from tangermeme.saturation_mutagenesis import saturation_mutagenesis
    except ImportError:
        from tangermeme.ism import saturation_mutagenesis
    import seaborn
    seaborn.set_style('whitegrid')
except ImportError as e:
    print(f"Failed to import necessary libraries. Please ensure 'tangermeme', 'captum', and 'tfbs' are installed (pip install '.[viz]').")
    print(f"Error: {e}")
    sys.exit(1)


# --- 1. Configuration and Utility Classes ---

class AnalysisConfig:
    """Handles command-line argument parsing and configuration storage."""
    _report_collector: Optional["ReportCollector"] = None

    def __init__(self):
        self.parser = self._setup_parser()
        self.args = self.parser.parse_args()
        self._validate_paths()

    def _setup_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Run genomic interpretability analysis on a trained model.",
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        # Required Arguments
        parser.add_argument('--model-path', type=str, required=True, 
                            help="Path to the model checkpoint (e.g., /path/to/best_model.ckpt).")
        parser.add_argument('--model-type', type=str, required=True, 
                            help="The name of the model class to load (e.g., 'BPNetReal' or 'RNNModel').")
        
        # Optional Arguments with Defaults
        parser.add_argument('--output-dir', type=str, default=None, 
                            help="Directory to save plots. If None, plots are displayed interactively.")
        parser.add_argument('--input-length', type=int, default=24, 
                            help="Sequence length used for the model input (default: 24).")
        parser.add_argument('--peak-length', type=int, default=24, 
                            help="Length of sequences to extract around ChIP-Seq peaks (default: 24).")
        parser.add_argument('--n-shuffles', type=int, default=20, 
                            help="Number of random shuffles for DeepLiftShap baselines (default: 20).")
        parser.add_argument('--peaks-file', type=str, 
                            default=os.environ.get("TFBS_PEAKS_BED", ""), 
                            help="Path to the ChIP-Seq peaks BED file for sequence extraction.")
        parser.add_argument('--jaspar-motif-file', type=str, 
                            default=os.environ.get("TFBS_MOTIF_FILE", ""), 
                            help="Path to the JASPAR core motifs MEME file.")
        parser.add_argument('--genome-fasta', type=str, 
                            default=os.environ.get("TFBS_GENOME_FASTA", ""), 
                            help="Path to the reference genome FASTA file.")
        parser.add_argument('--test', action='store_true', 
                            help="If set, runs a quick test with fewer sequences for faster execution.")
        parser.add_argument('--tangermeme', action='store_true', 
                            help="If set, uses Tangermeme's DeepLiftShap implementation instead of Captum's.")
        # New CLI arg to pick analyses
        parser.add_argument('--analyses', type=str, default='all',
                    help="Comma-separated list of analyses to run. Options: sequence,ambiguous_sequence,tad_scan,experimental_sequences,marginalization,seqlet,mutagenesis,rc,epistasis,epistasis_3d,comparison,conv_filters. Default: all")
        parser.add_argument('--max-seqs', type=int, default=100,
                            help="Maximum number of sequences to process for genome-wide analyses (default: 100).")
        parser.add_argument('--html-report', action='store_true',
                            help="If set and --output-dir provided, generate a simple HTML report collecting saved images.")
        return parser

    def _validate_paths(self):
        """Warns about data paths that are unset or point somewhere unusable."""
        data_paths = {
            '--peaks-file': 'TFBS_PEAKS_BED',
            '--jaspar-motif-file': 'TFBS_MOTIF_FILE',
            '--genome-fasta': 'TFBS_GENOME_FASTA',
        }

        for arg_name, env_var in data_paths.items():
            arg_val = getattr(self.args, arg_name.lstrip('--').replace('-', '_')) or ''
            if not arg_val:
                warnings.warn(
                    f"No path given for {arg_name}. Pass it explicitly or set "
                    f"the {env_var} environment variable.",
                    UserWarning
                )
            elif arg_val.startswith('/sc-projects/'):
                warnings.warn(
                    f"The path for {arg_name} points at an HPC cluster share: {arg_val}. "
                    "Verify it is correct for your setup.",
                    UserWarning
                )
            elif not os.path.exists(arg_val):
                warnings.warn(
                    f"The path for {arg_name} does not exist: {arg_val}",
                    UserWarning
                )
        
        if self.args.output_dir:
            os.makedirs(self.args.output_dir, exist_ok=True)
            print(f"Plots will be saved to: {self.args.output_dir}")
        else:
            print("No output directory provided. Plots will be displayed interactively.")

    def __getattr__(self, name):
        """Allows config.attr access instead of config.args.attr."""
        return getattr(self.args, name)

def save_or_show_plot(filename: str, output_dir: Optional[str], subfolder: Optional[str] = None):
    """Saves the current Matplotlib plot or displays it interactively.
       Also records saved images for the HTML report if enabled.

       Args:
           subfolder: Optional subfolder within output_dir (e.g. "deeplift").
    """
    if output_dir and filename:
        target_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
        os.makedirs(target_dir, exist_ok=True)
        safe_filename = filename.replace(' ', '_').replace('/', '_').replace(':', '')
        full_path = os.path.join(target_dir, safe_filename)

        plt.savefig(full_path, bbox_inches='tight')
        print(f"Plot saved to: {full_path}")
        # record for report
        if hasattr(AnalysisConfig, "_report_collector") and AnalysisConfig._report_collector:
            AnalysisConfig._report_collector.record(full_path)
    else:
        plt.show()

    plt.close()

# --- New: simple HTML report collector ---
class ReportCollector:
    def __init__(self, output_dir: Optional[str]):
        self.output_dir = output_dir
        self.saved_files: List[str] = []

    def record(self, path: str):
        self.saved_files.append(path)

    def write_html_report(self, title: str = "Genomic Interpreter Report"):
        if not self.output_dir or not self.saved_files:
            return
        p = pathlib.Path(self.output_dir)
        html_lines = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'><title>{}</title></head><body>".format(html.escape(title)),
            f"<h1>{html.escape(title)}</h1>",
            "<ul>"
        ]
        for f in self.saved_files:
            rel = pathlib.Path(f).relative_to(p)
            html_lines.append(f"<li><img src='{html.escape(str(rel))}' style='max-width:100%;height:auto'/></li>")
        html_lines.append("</ul></body></html>")
        out_path = os.path.join(self.output_dir, "report_index.html")
        with open(out_path, "w") as fh:
            fh.write("\n".join(html_lines))
        print(f"HTML report written to: {out_path}")

# --- 2. Model and Core Attribution Logic ---

class ModelLoader:
    """Handles dynamic loading and setup of the PyTorch model."""
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        print(f"Model loaded successfully on device: {self.device}")
        
    def _load_model(self) -> torch.nn.Module:
        """Loads the model dynamically from the checkpoint."""
        print(f"Loading model '{self.config.model_type}' from: {self.config.model_path}")
        try:
            model_cls = getattr(models, self.config.model_type)
            print(f"-> Found model class: {model_cls.__name__}")
            
            model = model_cls.load_from_checkpoint(self.config.model_path)
            
        except AttributeError:
            raise RuntimeError(f"Model type '{self.config.model_type}' not found in 'tfbs.nn.models'.")
        except FileNotFoundError:
            raise RuntimeError(f"Model checkpoint not found at {self.config.model_path}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during model loading: {e}")
            
        print("-> Model architecture:")
        print(model)
                
        model.eval()
        if hasattr(model, "freeze"):
            model.freeze()
        
        return model.to(self.device)


class AttributionCore:
    """
    Handles gradient-based attribution methods.
    
    CRITICAL FIX: Implements the cuDNN RNN backward pass fix by temporarily 
    switching the model to .train() mode for gradient calculation.
    """
    def __init__(self, model: torch.nn.Module, device: torch.device, tangermeme: bool):
        self.model = model
        self.device = device
        self.tangermeme = tangermeme

    def run_deep_lift_shap(self, X_input: torch.Tensor, n_shuffles: int, random_state: int = 0, verbose: bool = True, raw_outputs: bool = False, hypothetical: bool = False) -> torch.Tensor:
        """
        Batched Captum DeepLiftShap with:
         - raw_outputs (return multipliers per-shuffle: (N, n_shuffles, C, L))
         - hypothetical (return attributions for all characters)
         - memory-safe batching to avoid GPU OOM
         - cuDNN RNN backward fix (temporary train() mode)
        """
        # If the user did not request the tangermeme backend keep the captum function
        if self.tangermeme:
            return tangermeme_deep_lift_shap(  # type: ignore[return-value]
                self.model, X_input, args=None, target=0, batch_size=128,
                n_shuffles=n_shuffles, return_references=False,
                hypothetical=hypothetical, warning_threshold=1000, raw_outputs=raw_outputs,
                device=str(self.device), random_state=random_state, verbose=verbose
            )

        if verbose:
            print(f"Calculating DeepLiftShap attributions with {n_shuffles} shuffles (batched Captum).")

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        device = self.device
        N, C, L = X_input.shape
        if N == 0:
            # nothing to do
            return torch.empty((0, C, L), dtype=X_input.dtype).to(device) if not raw_outputs else torch.empty((0, n_shuffles, C, L), dtype=X_input.dtype)

        # tuneable parameter to control memory usage: max number of (input,baseline) pairs per call
        max_pairs = 256

        # result containers on CPU to reduce GPU pressure
        result_raw = torch.zeros((N, n_shuffles, C, L), dtype=X_input.dtype, device='cpu')
        result_sum = torch.zeros((N, C, L), dtype=X_input.dtype, device='cpu')

        i = 0
        while i < N:
            # choose a batch_N such that batch_N * 1 <= max_pairs (we will control shuffles per call)
            batch_N = min(N - i, max_pairs)
            # choose shuffles per call so batch_N * s <= max_pairs
            shuffles_per_call = max(1, max_pairs // batch_N)
            shuffles_per_call = min(shuffles_per_call, n_shuffles)

            X_batch = X_input[i:i + batch_N]  # keep on device

            s0 = 0
            while s0 < n_shuffles:
                s = min(shuffles_per_call, n_shuffles - s0)

                # generate s baselines and tile per input in batch
                baseline_block = random_one_hot((s, C, L)).float().to(device)  # (s, C, L)
                # create (batch_N * s, C, L) baselines where each baseline_block repeats for each example
                baselines_per_input = torch.cat([baseline_block for _ in range(batch_N)], dim=0).to(device)

                # repeat each input s times so ordering matches baselines_per_input
                X_batch_repeated = X_batch.repeat_interleave(s, dim=0).to(device)

                # decide custom attribution function for Captum
                custom_attr_func = None if raw_outputs else hypothetical_attributions

                # compute attributions for this chunk with train-mode toggle for RNNs
                was_training = self.model.training
                self.model.train()
                try:
                    dl_shap = DeepLiftShap(self.model)
                    a_chunk = dl_shap.attribute(
                        X_batch_repeated,
                        baselines=baselines_per_input,
                        target=0,
                        return_convergence_delta=False,
                        custom_attribution_func=custom_attr_func
                    )
                finally:
                    if not was_training:
                        self.model.eval()

                # a_chunk is on device; handle shapes and semantics

                # RAW outputs requested: store per-shuffle multipliers
                if raw_outputs:
                    if a_chunk.dim() == 3 and a_chunk.shape[0] == batch_N * s:
                        # (batch_N * s, C, L) -> (batch_N, s, C, L)
                        a_cpu = a_chunk.cpu().detach().reshape(batch_N, s, C, L)
                    elif a_chunk.dim() == 4 and a_chunk.shape == (batch_N, s, C, L):
                        a_cpu = a_chunk.cpu().detach()
                    elif a_chunk.dim() == 3 and a_chunk.shape == (batch_N, C, L):
                        # Captum returned averaged chunk; replicate across s
                        a_cpu = a_chunk.cpu().detach().unsqueeze(1).expand(batch_N, s, C, L)
                    else:
                        raise RuntimeError(f"Unexpected attribution chunk shape {tuple(a_chunk.shape)} for raw_outputs, batch_N={batch_N}, s={s}")

                    # place into result_raw in correct shuffle window
                    result_raw[i:i + batch_N, s0:s0 + s] = a_cpu.to('cpu')
                else:
                    # Processed outputs: sum multipliers/hypothetical attributions across shuffles.
                    # The final multiplication by input (if not hypothetical) happens at the end.
                    if a_chunk.dim() == 3 and a_chunk.shape[0] == batch_N * s:
                        # (batch_N * s, C, L) -> (batch_N, s, C, L) -> sum over s -> (batch_N, C, L)
                        sum_for_chunk = a_chunk.reshape(batch_N, s, C, L).sum(dim=1)
                    elif a_chunk.dim() == 4 and a_chunk.shape == (batch_N, s, C, L):
                        # Already in (batch_N, s, C, L) format.
                        sum_for_chunk = a_chunk.sum(dim=1)
                    elif a_chunk.dim() == 3 and a_chunk.shape == (batch_N, C, L):
                        # Captum may have already averaged over baselines; we need the sum.
                        sum_for_chunk = a_chunk * s
                    else:
                        raise RuntimeError(f"Unexpected attribution chunk shape {tuple(a_chunk.shape)} for processed outputs, batch_N={batch_N}, s={s}")

                    # accumulate sum on CPU
                    result_sum[i:i + batch_N] += sum_for_chunk.cpu().detach()
                s0 += s
            i += batch_N

        # finalize return
        if raw_outputs:
            if verbose:
                print("DeepLiftShap calculation complete. Returning raw multipliers (N, n_shuffles, C, L).")
            return result_raw
        else:
            # 1. Average the summed attributions.
            avg_attributions = result_sum / n_shuffles

            # 2. Apply final input multiplication if not hypothetical, and move to device.
            if not hypothetical:
                # Move averages to device and multiply element-wise with original input.
                final_attributions = avg_attributions.to(device) * X_input
            else:
                final_attributions = avg_attributions.to(device)

            if verbose:
                print("DeepLiftShap calculation complete. Returning (N, C, L) attributions.")
            return final_attributions

    def calculate_input_x_gradient(self, X_input: torch.Tensor) -> torch.Tensor:
        """Computes InputXGradient attributions with the RNN backward fix."""
        print("Calculating InputXGradient attributions...")
        ixg = InputXGradient(self.model)
        
        # --- RNN FIX: Temporarily set model to training mode ---
        self.model.train()
        try:
            attributions = ixg.attribute(X_input, target=0)
        finally:
            # --- RNN FIX: Restore to evaluation mode ---
            self.model.eval()

        print("InputXGradient calculation complete.")
        return attributions

# --- 3. Data Preparation ---

class DataPreparer:
    """Utility class for generating and loading test sequences."""
    def __init__(self, input_length: int, device: torch.device):
        self.input_length = input_length
        self.device = device

    def get_ambiguous_sequence(self) -> str:
        """Returns the built-in ambiguous sequence template used for expansion analyses."""
        return "CACTGNNNNNNNNNGTGTTCTTGA"

    def get_tad_region(self) -> Tuple[str, int, int, str]:
        """Returns the built-in genomic region used for the TAD scan analysis."""
        return "chr14", 36000000, 37250000, "LARGE_TAD"

    def fetch_genomic_region_sequence(self, genome_fasta: str, chrom: str, start: int, end: int) -> str:
        """Fetches a genomic interval from the reference genome as an uppercase DNA string."""
        from pyfaidx import Fasta

        genome = Fasta(genome_fasta, as_raw=True, sequence_always_upper=True)
        return str(genome[chrom][start:end])

    def split_sequence_into_chunks(self, sequence: str, chunk_size: int, step_size: int = 1) -> List[Tuple[int, str]]:
        """Splits a sequence into sliding chunks of fixed size."""
        chunks = []
        for chunk_start in range(0, len(sequence) - chunk_size + 1, step_size):
            chunk_end = chunk_start + chunk_size
            chunks.append((chunk_start, sequence[chunk_start:chunk_end]))
        return chunks

    def get_consensus_sequence(self) -> str:
        """Returns a fixed consensus sequence truncated to input_length."""
        original_consensus_seqs = {
            70: "TCGTCGGCAGCGTCAGATGTGTATCACTGCCCAGTCAAGTGTTCTTGATCACCGCTCCGACTGCAGAAAA",
            24: "CACTGCCCAGTCAAGTGTTCTTGA"
        }
        
        # 1. Try to get the sequence that exactly matches the input_length
        consensus_seq = original_consensus_seqs.get(self.input_length)
        
        if consensus_seq:
            # 2. If an exact match is found (e.g., for 70 or 24), return it directly
            return consensus_seq
        else:
            # 3. If no exact match, use the longest sequence (70-mer) as the default source
            #    and truncate it to the requested length.
            #    This covers all other requested lengths (e.g., 60, 50, 20, etc.).
            default_long_seq = original_consensus_seqs.get(70)
            
            # Check if the 70-mer exists before trying to slice it
            if default_long_seq:
                # Truncate the long sequence to the desired input_length using the middle portion
                start = (len(default_long_seq) - self.input_length) // 2
                return default_long_seq[start:start + self.input_length]
            else:
                raise ValueError("Consensus sequence for length 70 is missing.")

    def expand_ambiguous_sequence(self, sequence: str) -> List[str]:
        """Expand a sequence containing N into all concrete A/C/G/T sequences."""
        cleaned_sequence = sequence.strip().upper()
        if len(cleaned_sequence) != self.input_length:
            raise ValueError(
                f"Ambiguous sequence length must match input_length ({self.input_length}); got {len(cleaned_sequence)}."
            )

        invalid_bases = sorted(set(re.sub(r"[ACGTN]", "", cleaned_sequence)))
        if invalid_bases:
            raise ValueError(
                f"Ambiguous sequence contains invalid characters: {''.join(invalid_bases)}. Only A, C, G, T, and N are allowed."
            )

        n_positions = [index for index, base in enumerate(cleaned_sequence) if base == 'N']
        if not n_positions:
            return [cleaned_sequence]

        concrete_sequences = []
        for replacement_bases in itertools.product('ACGT', repeat=len(n_positions)):
            seq_list = list(cleaned_sequence)
            for position, base in zip(n_positions, replacement_bases):
                seq_list[position] = base
            concrete_sequences.append(''.join(seq_list))

        return concrete_sequences

    def load_promotor_sequences(self) -> Tuple[torch.Tensor, List[str]]:
        """Loads the collection of test sequences and names."""
        consensus_seq = self.get_consensus_sequence()
        promotor_sequences_map = {
            "Thyroglobulin": "CACTGCCCAGTCAAGTGTTCTTGA",
            "TPO_1": "ATGCCCACTCAAGCTT",
            "TPO_2": "CTATGAGTGGCACCT",
            "TPO_3": "ACAAATACTAAACAAACAGAATGG",
            "TSHR": "GGAGAGGCTCTCAAGTGCTCCGCTTGCC",
            "NIS": "AGACACGAGTGTTCCCCCACCCCGACTGCCCGCACCCCTG",
            "HHEX_1": "CGGGGGTTAGTGGGGGG",
            "HHEX_2": "CGGCAGGAAGGGGACCG",
            "SFTPA1": "GTGCTCCCCTCAAGGGTCCTA",
            "SFTPB": "TCAAAACACCTGGAGGGCTCTCCAGGACAAAG",
            "SFTPC": "TAGGCCAAGGGCTTGGGGGCTCT",
            "Nestin": "CTCCCAGAGGATGAGGTCATCGGCCTTGGCCTTGGGTGGG",
            "SHH": "ACAGAAATTGTTTTTTAAGTAGTTGCCCTCTGAAAATAT"
        }

        X_promotor = []
        test_names = []
        center_pos = self.input_length // 2 
        
        # Process known promoter sequences
        for name, seq in promotor_sequences_map.items():
            seq_len = len(seq)
            flank_half = seq_len // 2
            mod_seq = list(consensus_seq)
            start = center_pos - flank_half
            end = start + seq_len
            
            if start < 0 or end > self.input_length:
                # print(f"Warning: Skipping {name}...")
                continue

            mod_seq[start:end] = list(seq)
            mod_seq = ''.join(mod_seq)
            X_promotor.append(one_hot_encode(mod_seq))
            test_names.append(name)

        if not X_promotor:
            # Fallback to random if no promoter sequences fit
            X_promotor.append(random_one_hot((1, 4, self.input_length)).float())
            test_names = ["Random Sequence"]

        X_promotor = torch.stack(X_promotor).float().to(self.device)

        return X_promotor, test_names

    def load_maria_sequences(self) -> torch.Tensor:
        """Loads a fixed set of sequences for Maria's analysis."""
        maria_seqs = """CACTGCCCAGTCAAGTGTTCTTGA
        CACTGCCCAGTTAAGTGTTCTTGA
        CACTGCCCAGTAAAGTGTTCTTGA
        CACTGCCCAGTGAAGTGTTCTTGA
        CACTGCCCAGTCGAGTGTTCTTGA
        CACTGCCCAGTCTAGTGTTCTTGA
        CACTGCCCAGTCCAGTGTTCTTGA
        CACTGCCCAGTCACGTGTTCTTGA
        CACTGCCCAGTCATGTGTTCTTGA
        CACTGCCCAGTCAGGTGTTCTTGA
        CACTGCCCAGTCAACTGTTCTTGA
        CACTGCCCAGTCAATTGTTCTTGA
        CACTGCCCAGTCAAATGTTCTTGA
        CACTGCCCAGTCCACTGTTCTTGA
        CACTGCCCAGTCCATTGTTCTTGA
        CACTGCCCAGTCCAATGTTCTTGA
        CACTGCCCAGTAAACTGTTCTTGA
        CACTGCCCAGACAAGTGTTCTTGA
        CACTGCCCAGCCAAGTGTTCTTGA
        CACTGCCCAGGCAAGTGTTCTTGA
        CACTGCCCAATCAAGTGTTCTTGA
        CACTGCCCACTCAAGTGTTCTTGA
        CACTGCCCATTCAAGTGTTCTTGA
        CACTGCCCAGTCAAGAGTTCTTGA
        CACTGCCCAGTCAAGCGTTCTTGA
        CACTGCCCAGTCAAGGGTTCTTGA
        CACTGCCCAGTCAAGTATTCTTGA
        CACTGCCCAGTCAAGTCTTCTTGA
        CACTGCCCAGTCAAGTTTTCTTGA
        CACTGCCCACTCAAGTGTTCTTGA
	    CACTGCCCAGTCACGTGCTCTTGA
	    CACTGCCCAGTCACGTGTTCTTGA
	    CACTGCCCAGTCATGTGTACTTGA
	    CACTGCCCAGTCATGTGTGCTTGA
	    CACTGCCCACTCAAGTGTTATTGA
	    CACTGCCCAGGCAAGTGTTCTTGA"""
        
        lines = [line.strip() for line in maria_seqs.splitlines() if line.strip()]
        seqs = []
        for line in lines: 
            if len(line) >= self.input_length:
                start = (len(line) - self.input_length) // 2
                truncated = line[start:start + self.input_length]
                seqs.append(truncated)
        
        if not seqs:
            print("Warning: No Maria sequences could be loaded. Returning empty tensor.")
            return torch.empty((0, 4, self.input_length), dtype=torch.float32).to(self.device)

        return torch.stack([one_hot_encode(seq) for seq in seqs]).float().to(self.device)
    
    def load_medium_maria_sequences(self) -> torch.Tensor:
        maria_seqs = """CCAGTCAAGTGTTC
        CCAGGCAAGTGTTC
        CCAGGCAAGTGACC
        CCAGTCAAGTGACC"""
        seqs = []

        for line in maria_seqs.splitlines():
            line = line.strip()
            seqs.append(line)

        if not seqs:
            print("Warning: No medium Maria sequences could be loaded. Returning empty tensor.")
            return torch.empty((0, 4, self.input_length), dtype=torch.float32).to(self.device)
        
        return torch.stack([one_hot_encode(seq) for seq in seqs]).float().to(self.device)

    def load_short_maria_sequences(self) -> torch.Tensor:
        """Loads a set of shorter sequences for Maria's analysis."""
        maria_seqs = """ACTCAAGTGTTC
        AGTCACGTGCTC
        AGTCACGTGTTC
        AGTCATGTGTAC
        AGTCATGTGTGC
        ACTCAAGTGTTA
        AGGCAAGTGTTC
        AGTCAAGTGTTC
        AGTCACGTGTTC"""
        
        seqs = []
        
        for line in maria_seqs.splitlines():
            line = line.strip()
            seqs.append(line)

        if not seqs:
            print("Warning: No short Maria sequences could be loaded. Returning empty tensor.")
            return torch.empty((0, 4, self.input_length), dtype=torch.float32).to(self.device)

        return torch.stack([one_hot_encode(seq) for seq in seqs]).float().to(self.device)
    

    def load_fabbro_sequences(self) -> torch.Tensor:
        """Loads Fabbro et al. sequences from a hardcoded list."""
        fabbro_seqs = { "S09":"CCACTCAAACGGCTTTT",
                        "S10":"CCAAGTCAAATGTTCCT",
                        "S11":"CCAACTCAAGTGGTATT",
                        "S16":"CCGACTCAAGGGCTATT",
                        "S21":"CCGACTCAAGGGCTCTT",
                        "S23":"CCAGCTCAAGTGGTCTT",
                        "S24":"CCGAGTCAAGTGGTGTT",
                        "S26":"CCACTGCAACGGCTTTT",
                        "S28":"CCACTTGAATTGCTGTT",
                        "S29":"CCGGGACAAGCGTTTTT",
                        "S30":"CCCACTGAAGTGCTCTT",
                        "S31":"CCGAATGAACTGGTGTT",
                        "S35":"CCTACCCAACTGCTCTT",
                        "S36":"CCGGCTCAAGGGTTCTT"}

        seqs = []

        for key, seq in fabbro_seqs.items():
            seqs.append(seq)

        if not seqs:
            print("Warning: No Fabbro sequences could be loaded. Returning empty tensor.")
            return torch.empty((0, 4, self.input_length), dtype=torch.float32).to(self.device)

        return torch.stack([one_hot_encode(seq) for seq in seqs]).float().to(self.device)

# --- 4. Main Orchestration Class ---

class GenomicInterpreter:
    """The main class orchestrating the entire interpretability workflow."""
    def __init__(self, model: torch.nn.Module, config: AnalysisConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.attr_core = AttributionCore(model, device, config.tangermeme)
        self.data_prep = DataPreparer(config.input_length, device)
        # ensure a shared report collector exists on AnalysisConfig for save_or_show_plot to find
        if config.output_dir and config.html_report:
            AnalysisConfig._report_collector = ReportCollector(config.output_dir)
        else:
            AnalysisConfig._report_collector = None
        
    def _plot_attributions(self, X_attr: torch.Tensor, names: List[str], prefix: str):
        """Utility to plot and save a set of attributions."""
        for i in range(X_attr.shape[0]):
            plt.figure(figsize=(10, 2))
            ax = plt.subplot(111)
            X_attr_np = X_attr[i].detach().cpu().numpy().astype(float)
            plot_logo(X_attr_np, ax=ax)
            plt.xlabel("Genomic Position")
            plt.ylabel("Attribution")
            # plt.ylim(-0.5, 0.5)
            seq_name = names[i]
            plt.title(f"{prefix} Attributions for {seq_name}")
            plt.tight_layout()
            save_or_show_plot(f"{prefix}_{seq_name}.png", self.config.output_dir, subfolder="deeplift")

    def run_all_analysis(self):
        """Runs the pipeline depending on requested analyses."""
        requested = [a.strip().lower() for a in self.config.analyses.split(',')] if isinstance(self.config.analyses, str) else self.config.analyses
        if 'all' in requested:
            order = [
                ('sequence', self.run_sequence_analysis),
                ('ambiguous_sequence', self.run_ambiguous_sequence_analysis),
                ('tad_scan', self.run_tad_scan_analysis),
                ('experimental_sequences', self.run_experimental_sequences_analysis),
                ('marginalization', self.run_marginalization_analysis),
                ('seqlet', self.run_seqlet_analysis),
                ('mutagenesis', self.run_mutagenesis_analysis),
                ('rc', self.run_rc_comparison),
                ('epistasis', self.run_epistasis_analysis),
                ('epistasis_3d', self.run_epistasis_3D_analysis),
                ('comparison', self.run_attribution_method_comparison),
                ('conv_filters', self.run_conv_filter_visualization) # NEW: Filter visualization
            ]
        else:
            mapping = {
                'sequence': self.run_sequence_analysis,
                'ambiguous_sequence': self.run_ambiguous_sequence_analysis,
                'tad_scan': self.run_tad_scan_analysis,
                'experimental_sequences': self.run_experimental_sequences_analysis,
                'marginalization': self.run_marginalization_analysis,
                'seqlet': self.run_seqlet_analysis,
                'mutagenesis': self.run_mutagenesis_analysis,
                'rc': self.run_rc_comparison,
                'epistasis': self.run_epistasis_analysis,
                'epistasis_3d': self.run_epistasis_3D_analysis,
                'comparison': self.run_attribution_method_comparison,
                'conv_filters': self.run_conv_filter_visualization # NEW: Filter visualization
            }
            order = [(name, mapping[name]) for name in requested if name in mapping]

        for name, func in order:
            print(f"\nRunning analysis: {name}")
            try:
                func()
            except Exception as e:
                print(f"Warning: Analysis '{name}' failed with error: {e}")

        # write HTML report if requested
        if self.config.output_dir and self.config.html_report and AnalysisConfig._report_collector:
            AnalysisConfig._report_collector.write_html_report()

    # --- Analysis Methods ---

    def run_sequence_analysis(self):
        """Analyzes promoter and Maria sequences using DeepLiftShap."""
        print("\n" + "="*50)
        print("--- 1. Promoter Sequence Analysis (DeepLiftShap) ---")
        
        X_promotor, names = self.data_prep.load_promotor_sequences()
        predictions = tpred(self.model, X_promotor, device=str(self.device))  # type: ignore[arg-type]
        # Save predictions to file if output_dir is specified
        if self.config.output_dir:
            pred_df = pd.DataFrame({
            "Sequence_Name": names,
            "Prediction": predictions.detach().cpu().numpy().flatten()  # type: ignore[union-attr]
            })
            pred_path = os.path.join(self.config.output_dir, "promoter_predictions.csv")
            pred_df.to_csv(pred_path, index=False)
            print(f"Promoter predictions saved to: {pred_path}")
        for i in range(predictions.shape[0]):  # type: ignore[union-attr]
            print(f"{names[i]}: Prediction = {predictions[i].item():.4f}")

        X_attr = self.attr_core.run_deep_lift_shap(X_promotor, self.config.n_shuffles, verbose=True)
        self._plot_attributions(X_attr, names, "DeepLiftShap")
        
        # # Maria sequences
        # print("\n" + "="*50)
        # print("--- 2. Maria Sequences Analysis (DeepLiftShap) ---")
        # X_maria = self.data_prep.load_maria_sequences()
        # print(f"Loaded {X_maria.shape[0]} Maria sequences for analysis.")
        # names = [f"Maria_Seq_{i+1}" for i in range(X_maria.shape[0])]
        # if X_maria.shape[0] > 0:
        #     predictions = tpred(self.model, X_maria, device=self.device)
        #     if self.config.output_dir:
        #         pred_df = pd.DataFrame({
        #         "Sequence_Name": names,
        #         "Prediction": predictions.detach().cpu().numpy().flatten()
        #         })
        #         pred_path = os.path.join(self.config.output_dir, "maria_predictions.csv")
        #         pred_df.to_csv(pred_path, index=False)
        #         print(f"Promoter predictions saved to: {pred_path}")
        #     for i in range(predictions.shape[0]):
        #         print(f"{names[i]}: Prediction = {predictions[i].item():.4f}")
        #     X_attr_maria = self.attr_core.run_deep_lift_shap(X_maria, self.config.n_shuffles, verbose=True)
        #     self._plot_attributions(X_attr_maria, names, "DeepLiftShap")
        # else:
        #     print("Skipping Maria sequence analysis (no sequences loaded).")

    def run_ambiguous_sequence_analysis(self):
        """Expands an N-containing sequence, predicts all variants, and plots DeepLiftShap for each one."""
        print("\n" + "="*50)
        print("--- 2. Ambiguous Sequence Expansion (Prediction + DeepLiftShap) ---")

        ambiguous_sequence = self.data_prep.get_ambiguous_sequence()
        print(f"Using built-in ambiguous sequence template: {ambiguous_sequence}")

        concrete_sequences = self.data_prep.expand_ambiguous_sequence(ambiguous_sequence)
        print(f"Expanded ambiguous sequence into {len(concrete_sequences)} concrete sequences.")

        sequence_records = []
        expanded_tensors = []

        for index, sequence in enumerate(concrete_sequences, start=1):
            sequence_name = f"Ambiguous_Seq_{index:03d}"
            X_sequence = one_hot_encode(sequence).unsqueeze(0).float().to(self.device)
            expanded_tensors.append(X_sequence)

            prediction = tpred(self.model, X_sequence, device=str(self.device)).detach().cpu().numpy().item()  # type: ignore[union-attr]
            sequence_records.append({
                "Sequence_Name": sequence_name,
                "Sequence": sequence,
                "Prediction": prediction,
            })
            print(f"{sequence_name}: {sequence} -> Prediction = {prediction:.4f}")

        if self.config.output_dir:
            pred_df = pd.DataFrame(sequence_records)
            pred_path = os.path.join(self.config.output_dir, "ambiguous_sequence_predictions.csv")
            pred_df.to_csv(pred_path, index=False)
            print(f"Ambiguous sequence predictions saved to: {pred_path}")

        X_expanded = torch.cat(expanded_tensors, dim=0)
        X_attr = self.attr_core.run_deep_lift_shap(X_expanded, self.config.n_shuffles, verbose=True)

        for index, record in enumerate(sequence_records):
            plt.figure(figsize=(10, 2))
            ax = plt.subplot(111)
            plot_logo(X_attr[index].detach().cpu().numpy().astype(float), ax=ax)
            plt.xlabel("Genomic Position")
            plt.ylabel("Attribution")
            plt.title(f"DeepLiftShap Attributions for {record['Sequence_Name']}")
            plt.tight_layout()
            save_or_show_plot(f"DeepLiftShap_{record['Sequence_Name']}.png", self.config.output_dir, subfolder="deeplift")

        # --- PWM comparison: top vs bottom 10% of predicted sequences ---
        self._run_ambiguous_pwm_comparison(sequence_records, ambiguous_sequence)

    def _run_ambiguous_pwm_comparison(self, sequence_records: List[dict], ambiguous_sequence: str):
        """Build PWMs from the top and bottom 10% of ambiguous sequence predictions and compare."""
        from logomaker import Logo, transform_matrix

        print("\n--- Ambiguous Sequence PWM Comparison (Top vs Bottom 10%) ---")

        df = pd.DataFrame(sequence_records)
        n_total = len(df)
        n_percentile = max(1, n_total // 10)

        df_sorted = df.sort_values("Prediction", ascending=False)
        top_seqs = df_sorted.head(n_percentile)["Sequence"].tolist()
        bottom_seqs = df_sorted.tail(n_percentile)["Sequence"].tolist()

        print(f"Total sequences: {n_total}, Top 10%: {len(top_seqs)} (score >= {df_sorted.iloc[n_percentile-1]['Prediction']:.4f}), "
              f"Bottom 10%: {len(bottom_seqs)} (score <= {df_sorted.iloc[-n_percentile]['Prediction']:.4f})")

        def seqs_to_pwm(sequences: List[str]) -> pd.DataFrame:
            """Convert a list of equal-length DNA sequences to a PWM (probability matrix)."""
            seq_len = len(sequences[0])
            counts = {base: [0] * seq_len for base in "ACGT"}
            for seq in sequences:
                for pos, base in enumerate(seq):
                    counts[base][pos] += 1
            pwm = pd.DataFrame(counts)
            pwm = pwm.div(pwm.sum(axis=1), axis=0)
            return pwm

        pwm_top = seqs_to_pwm(top_seqs)
        pwm_bottom = seqs_to_pwm(bottom_seqs)
        pwm_diff = pwm_top - pwm_bottom

        # Identify ambiguous (N) positions
        n_positions = [i for i, base in enumerate(ambiguous_sequence) if base == 'N']

        # Save PWMs to CSV
        if self.config.output_dir:
            pwm_top.to_csv(os.path.join(self.config.output_dir, "ambiguous_pwm_top10.csv"), index_label="Position")
            pwm_bottom.to_csv(os.path.join(self.config.output_dir, "ambiguous_pwm_bottom10.csv"), index_label="Position")
            pwm_diff.to_csv(os.path.join(self.config.output_dir, "ambiguous_pwm_diff.csv"), index_label="Position")

        # --- Plot: Top vs Bottom PWMs side by side ---
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True)

        # Information content logos (bits) for top and bottom
        info_top = transform_matrix(pwm_top, from_type="probability", to_type="information")
        info_bottom = transform_matrix(pwm_bottom, from_type="probability", to_type="information")

        Logo(info_top, ax=axes[0], color_scheme="classic")
        axes[0].set_title(f"Top 10% sequences (n={len(top_seqs)}, mean score={top_seqs and df_sorted.head(n_percentile)['Prediction'].mean():.4f})")
        axes[0].set_ylabel("Bits")
        for pos in n_positions:
            axes[0].axvspan(pos - 0.5, pos + 0.5, alpha=0.1, color="grey")

        Logo(info_bottom, ax=axes[1], color_scheme="classic")
        axes[1].set_title(f"Bottom 10% sequences (n={len(bottom_seqs)}, mean score={bottom_seqs and df_sorted.tail(n_percentile)['Prediction'].mean():.4f})")
        axes[1].set_ylabel("Bits")
        for pos in n_positions:
            axes[1].axvspan(pos - 0.5, pos + 0.5, alpha=0.1, color="grey")

        # Differential logo (top - bottom)
        Logo(pwm_diff, ax=axes[2], color_scheme="classic")
        axes[2].set_title("Differential PWM (Top 10% − Bottom 10%)")
        axes[2].set_ylabel("Δ Frequency")
        axes[2].set_xlabel("Position")
        axes[2].axhline(y=0, color='black', linewidth=0.5, linestyle='-')
        for pos in n_positions:
            axes[2].axvspan(pos - 0.5, pos + 0.5, alpha=0.1, color="grey")

        save_or_show_plot("Ambiguous_PWM_Comparison.png", self.config.output_dir)

    def run_tad_scan_analysis(self):
        """Scans the TAD region with a moving window and writes predictions + summary heatmap."""
        print("\n" + "="*50)
        print("--- TAD Region Scan (Moving Window Prediction + DeepLiftShap) ---")

        if not self.config.output_dir:
            print("Skipping TAD scan because --output-dir was not provided. This analysis writes a summary figure.")
            return

        tad_dir = self.config.output_dir

        chrom, region_start, region_end, region_name = self.data_prep.get_tad_region()
        region_sequence = self.data_prep.fetch_genomic_region_sequence(
            self.config.genome_fasta,
            chrom,
            region_start,
            region_end,
        )

        window_size = 50
        step_size = 25
        chunks = self.data_prep.split_sequence_into_chunks(region_sequence, window_size, step_size=step_size)

        print(
            f"Loaded {region_name}: {chrom}:{region_start}-{region_end} ({len(region_sequence)} bp). "
            f"Generated {len(chunks)} windows ({window_size} bp, step {step_size})."
        )

        if not chunks:
            print(f"No full {window_size}-mer windows could be generated from the TAD region.")
            return

        batch_size = 256
        prediction_records = []
        attribution_rows: List[np.ndarray] = []
        window_starts: List[int] = []
        prediction_values: List[float] = []

        for batch_start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[batch_start:batch_start + batch_size]
            batch_sequences = [sequence for _, sequence in batch_chunks]

            X_batch = torch.stack([one_hot_encode(sequence) for sequence in batch_sequences]).float().to(self.device)
            batch_predictions = tpred(self.model, X_batch, device=str(self.device)).detach().cpu().view(-1).numpy()  # type: ignore[union-attr]

            batch_attributions = self.attr_core.run_deep_lift_shap(X_batch, self.config.n_shuffles, verbose=False)

            for local_index, (offset, sequence) in enumerate(batch_chunks):
                genomic_start = region_start + offset
                genomic_end = genomic_start + window_size
                prediction_value = float(batch_predictions[local_index])

                prediction_records.append({
                    "Region": region_name,
                    "Chromosome": chrom,
                    "Start": genomic_start,
                    "End": genomic_end,
                    "Sequence": sequence,
                    "Prediction": prediction_value,
                })
                attribution_rows.append(batch_attributions[local_index].detach().cpu().numpy().astype(float).sum(axis=0))
                window_starts.append(genomic_start)
                prediction_values.append(prediction_value)

            print(f"Processed {min(batch_start + batch_size, len(chunks))}/{len(chunks)} TAD windows.")

        prediction_df = pd.DataFrame(prediction_records)
        prediction_path = os.path.join(self.config.output_dir, "tad_predictions.csv")
        prediction_df.to_csv(prediction_path, index=False)
        print(f"TAD predictions saved to: {prediction_path}")

        attribution_matrix = np.vstack(attribution_rows)
        heatmap_path = os.path.join(tad_dir, "tad_summary_heatmap.png")

        prediction_matrix = np.asarray(prediction_values, dtype=float)[np.newaxis, :]

        fig, (ax_pred, ax_heat) = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)

        if prediction_matrix.size:
            pred_min = float(np.min(prediction_matrix))
            pred_max = float(np.max(prediction_matrix))
            pred_center = float(np.median(prediction_matrix))
            pred_norm = None if pred_min == pred_max else TwoSlopeNorm(vmin=pred_min, vcenter=pred_center, vmax=pred_max)
        else:
            pred_norm = None

        pred_im = ax_pred.imshow(
            prediction_matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            norm=pred_norm,
            interpolation="nearest",
        )
        ax_pred.set_title(f"{region_name} prediction intensity ({window_size} bp windows, step {step_size})")
        ax_pred.set_ylabel("Prediction")
        ax_pred.set_yticks([])

        max_abs_attr = float(np.max(np.abs(attribution_matrix))) if attribution_matrix.size else 1.0
        if max_abs_attr == 0:
            max_abs_attr = 1.0
        norm = TwoSlopeNorm(vmin=-max_abs_attr, vcenter=0, vmax=max_abs_attr)

        im = ax_heat.imshow(
            attribution_matrix,
            aspect="auto",
            origin="lower",
            cmap="coolwarm",
            norm=norm,
            interpolation="nearest",
        )
        ax_heat.set_title(f"{region_name} signed DeepLiftShap summary heatmap")
        ax_heat.set_xlabel(f"Position within {window_size}-mer")
        ax_heat.set_ylabel("Window start position")

        x_tick_positions = np.arange(0, window_size, 5)
        ax_heat.set_xticks(x_tick_positions)
        ax_heat.set_xticklabels([str(pos + 1) for pos in x_tick_positions])

        if len(window_starts) > 10:
            y_tick_positions = np.linspace(0, len(window_starts) - 1, 8, dtype=int)
        else:
            y_tick_positions = np.arange(len(window_starts))
        ax_heat.set_yticks(y_tick_positions)
        ax_heat.set_yticklabels([str(window_starts[pos]) for pos in y_tick_positions])

        pred_cbar = fig.colorbar(pred_im, ax=ax_pred, fraction=0.025, pad=0.04)
        pred_cbar.set_label("Prediction intensity")

        attr_cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.04)
        attr_cbar.set_label("Signed summed DeepLiftShap attribution")

        plt.savefig(heatmap_path, bbox_inches="tight")
        print(f"TAD summary heatmap saved to: {heatmap_path}")
        plt.close(fig)

    def _calculate_receptive_field(self, cap_to_input_length: bool = True) -> int:
        """Calculate the model's receptive field from its conv layers."""
        model_obj = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model_obj, 'conv_layers'):
            rf = 1
            for layer in model_obj.conv_layers:
                if isinstance(layer, torch.nn.Conv1d):
                    k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
                    d = layer.dilation[0] if isinstance(layer.dilation, tuple) else layer.dilation
                    rf += (k - 1) * d
            target = min(rf, self.config.input_length) if cap_to_input_length else rf
            print(f"Calculated Model Receptive Field: {rf} bp" + (f" (capped to {target} for plotting)" if cap_to_input_length else ""))
            return target
        print(f"Warning: Could not determine RF from model layers. Using fallback {self.config.input_length}bp.")
        return self.config.input_length

    def _pad_sequences(self, X: torch.Tensor, target_len: int) -> torch.Tensor:
        """Pad one-hot encoded sequences to target_len with zeros (N bases)."""
        current_len = X.shape[-1]
        if X.shape[0] > 0 and current_len < target_len:
            pad_amount = target_len - current_len
            print(f"Padding sequences with {pad_amount} 'N' bases (Current: {current_len}, Target: {target_len})...")
            if isinstance(X, torch.Tensor):
                padding = torch.full((X.shape[0], 4, pad_amount), 0, dtype=X.dtype, device=X.device)
                X = torch.cat([X, padding], dim=2)
            else:
                padding = np.full((X.shape[0], 4, pad_amount), 0, dtype=X.dtype)
                X = np.concatenate([X, padding], axis=2)
        return X

    def _run_sequence_set(self, X: torch.Tensor, label: str, prefix: str, csv_name: str):
        """Run prediction + DeepLiftShap attribution on a set of sequences."""
        print(f"Loaded {X.shape[0]} {label} sequences for analysis.")
        names = [f"{prefix}_{i+1}" for i in range(X.shape[0])]
        if X.shape[0] == 0:
            print(f"Skipping {label} analysis (no sequences loaded).")
            return
        predictions = tpred(self.model, X, device=str(self.device))  # type: ignore[arg-type]
        if self.config.output_dir:
            pred_df = pd.DataFrame({
                "Sequence_Name": names,
                "Prediction": predictions.detach().cpu().numpy().flatten()  # type: ignore[union-attr]
            })
            pred_path = os.path.join(self.config.output_dir, csv_name)
            pred_df.to_csv(pred_path, index=False)
            print(f"{label} predictions saved to: {pred_path}")
        for i in range(predictions.shape[0]):  # type: ignore[union-attr]
            print(f"{names[i]}: Prediction = {predictions[i].item():.4f}")
        X_attr = self.attr_core.run_deep_lift_shap(X, self.config.n_shuffles, verbose=True)
        self._plot_attributions(X_attr, names, "DeepLiftShap")

    def run_experimental_sequences_analysis(self):
        """Runs prediction and attribution on medium/short Maria and Fabbro sequences."""
        print("\n" + "="*50)
        print("--- Experimental Sequences Analysis (DeepLiftShap) ---")

        target_len = self._calculate_receptive_field(cap_to_input_length=True)

        # Medium Maria sequences
        print("\n--- Medium Maria Sequences ---")
        X_medium_maria = self.data_prep.load_medium_maria_sequences()
        X_medium_maria = self._pad_sequences(X_medium_maria, target_len)
        self._run_sequence_set(X_medium_maria, "medium Maria", "Medium_Maria_Seq", "medium_maria_predictions.csv")

        # Fabbro sequences
        print("\n--- Fabbro Sequences ---")
        X_fabbro = self.data_prep.load_fabbro_sequences()
        self._run_sequence_set(X_fabbro, "Fabbro", "Fabbro_Seq", "fabbro_predictions.csv")

        # Short Maria sequences
        print("\n--- Short Maria Sequences ---")
        X_short_maria = self.data_prep.load_short_maria_sequences()
        X_short_maria = self._pad_sequences(X_short_maria, self._calculate_receptive_field(cap_to_input_length=False))
        self._run_sequence_set(X_short_maria, "short Maria", "Short_Maria_Seq", "short_maria_predictions.csv")

    def run_marginalization_analysis(self):
        """Performs motif marginalization for prediction and attribution."""
        print("\n" + "="*50)
        print("--- 3. Marginalization Analysis ---")
        
        motifs = {}
        try:
            motifs.update(read_meme(self.config.jaspar_motif_file, n_motifs=10))
        except FileNotFoundError as e:
            print(f"Warning: Motif files not found. Skipping marginalization. ({e})")
            return
            
        motifs = {k: v for k, v in motifs.items() if v.shape[1] <= self.config.input_length}
        print(f"Loaded and filtered {len(motifs)} motifs.")

        X_marginal_test = random_one_hot((5, 4, self.config.input_length)).float().to(self.device)

        print("\nPrediction Marginalization (Delta Prediction):")
        for name, pwm in motifs.items():
            consensus = pwm_consensus(pwm).unsqueeze(0).to(self.device)
            y_before, y_after = marginalize(self.model, X_marginal_test, consensus, device=str(self.device))
            delta = (y_after - y_before).mean().item()  # type: ignore[operator]
            print(f"{name}: {delta:.4f}")

        print("\nAttribution Marginalization (Delta Attribution * Input):")
        for name, pwm in motifs.items():
            consensus = pwm_consensus(pwm).unsqueeze(0).to(self.device)
            
            # Use the AttributionCore method to ensure mode switching for the backward pass
            def dlshap_wrapper(model, X_input, **kwargs):
                # Ensure X_input is a torch tensor on the interpreter device
                if isinstance(X_input, np.ndarray):
                    X_in_t = torch.from_numpy(X_input).float()
                elif isinstance(X_input, torch.Tensor):
                    X_in_t = X_input
                else:
                    X_in_t = torch.as_tensor(X_input).float()
                if X_in_t.device != self.device:
                    X_in_t = X_in_t.to(self.device)
                # run attribution and ensure result is on the same device
                attr = self.attr_core.run_deep_lift_shap(X_in_t, n_shuffles=5, verbose=False)
                if not isinstance(attr, torch.Tensor):
                    attr = torch.as_tensor(attr)
                return attr.to(self.device)

            # marginalize may return numpy or torch tensors; coerce to torch on the correct device
            y_before_attr, y_after_attr = marginalize(self.model, X_marginal_test, consensus, func=dlshap_wrapper)

            if not isinstance(y_before_attr, torch.Tensor):
                y_before_attr = torch.as_tensor(y_before_attr)
            if not isinstance(y_after_attr, torch.Tensor):
                y_after_attr = torch.as_tensor(y_after_attr)

            # move attributions to the same device as X_marginal_test before arithmetic
            y_before_attr = y_before_attr.to(self.device)
            y_after_attr = y_after_attr.to(self.device)

            y_before_contrib = y_before_attr * X_marginal_test
            y_after_contrib = y_after_attr * X_marginal_test

            delta_contrib = (y_after_contrib - y_before_contrib).sum(dim=(1, 2)).mean().item()
            print(f"{name}: {delta_contrib:.4f}")

    def run_seqlet_analysis(self):
        """Performs genome-wide attribution, seqlet discovery, and motif enrichment."""
        print("\n" + "="*50)
        print("--- 4. Genome-wide Attribution and Seqlet Analysis ---")

        
        # Load sequences from peaks file if paths are valid
        peaks = pd.read_csv(self.config.peaks_file, sep="\t", usecols=(0, 1, 2), names=['chrom', 'start', 'end'])
        X_peaks = extract_loci(peaks, self.config.genome_fasta, in_window=self.config.peak_length).float()  # type: ignore[union-attr]
        X_peaks = X_peaks[X_peaks.sum(dim=(1, 2)) == self.config.peak_length].to(self.device) 
        X_analysis = X_peaks[:min(10 if self.config.test else self.config.max_seqs, X_peaks.shape[0])]
        print(f"Using {X_analysis.shape[0]} sequences extracted from peaks.")


        X_attr_analysis = self.attr_core.run_deep_lift_shap(X_analysis, self.config.n_shuffles, verbose=False)
        #X_attr_analysis = tangermeme_deep_lift_shap(self.model, X_analysis, batch_size=128, n_shuffles=self.config.n_shuffles, random_state=0, verbose=False, device=self.device, warning_threshold=1000)
        X_attr_sum = X_attr_analysis.sum(dim=1)

        if X_attr_sum.device != 'cpu':
            print(f"Moving X_attr_sum from {X_attr_sum.device} to CPU for seqlet analysis.")
            X_attr_sum_cpu = X_attr_sum.cpu().detach().numpy()
        else:
            X_attr_sum_cpu = X_attr_sum

        seqlets = recursive_seqlets(
            X_attr_sum_cpu, # Use the CPU tensor here
            additional_flanks=0, 
            min_seqlet_len=5, 
            max_seqlet_len=15
        )
        
        print(f"Successfully extracted {len(seqlets)} seqlets.")
        if seqlets.empty:
            print("No seqlets found. Skipping annotation and plotting.")
            return
        try:
            print(f"Annotating seqlets using motif file: {self.config.jaspar_motif_file}")
            motif_result = annotate_seqlets(X_analysis.cpu(), seqlets, self.config.jaspar_motif_file)
            motif_idxs = motif_result[0][:, 0]

            y_counts = count_annotations((seqlets['example_idx'], motif_idxs))
            motif_all = read_meme(self.config.jaspar_motif_file)
            motif_names = np.array(list(motif_all.keys()))

            y_sum = y_counts.sum(dim=0)
            idxs = torch.flip(torch.argsort(y_sum), dims=(-1,))

            names = motif_names[idxs[:10]]
            counts = y_sum[idxs[:10]].detach().cpu().numpy()
            print("\nTop 10 Enriched Motifs in Attributed Seqlets:")
            for name, count in zip(names, counts):
                print(f"  {name}: {int(count)} hits")

            print("\nPlotting annotated seqlets for the first 10 sequences...")

            # Collect all seqlets for the first 10 unique example indices
            if X_analysis.shape[0] > 0 and not seqlets.empty:
                unique_idxs = seqlets['example_idx'].unique()[:10]
                fig, axes = plt.subplots(len(unique_idxs), 1, figsize=(12, 2.5 * len(unique_idxs)), sharex=True)
                if len(unique_idxs) == 1:
                    axes = [axes]  # Ensure axes is iterable

                center = self.config.input_length // 2
                plot_window = min(20, self.config.input_length // 2)
                plot_start, plot_end = max(0, center - plot_window), min(self.config.input_length, center + plot_window)

                for ax, idx in zip(axes, unique_idxs):
                    idxs_plot = seqlets['example_idx'] == idx
                    seqlets_ = seqlets[idxs_plot].copy()
                    seqlets_.example_idx = seqlets_.example_idx.astype(object)
                    seqlets_.loc[:, 'example_idx'] = motif_names[motif_idxs[idxs_plot]]
                    seqlets_ = seqlets_[seqlets_['attribution'] > 0]

                    plot_logo(
                        X_attr_analysis[idx].detach().cpu().numpy().astype(float),
                        ax=ax, start=plot_start, end=plot_end,
                        annotations=seqlets_, score_key='attribution'
                    )
                    ax.set_ylabel("DeepLiftShap Attributions")
                    ax.set_title(f"Seqlet Annotation Example (Sequence {idx})")

                plt.xlabel("Genomic Coordinate")
                plt.tight_layout()
                    
                save_or_show_plot("Seqlet_Annotation_Examples.png", self.config.output_dir)

        except FileNotFoundError:
            print(f"Warning: JASPAR Motif MEME file not found at {self.config.jaspar_motif_file}. Skipping seqlet annotation and FIMO.")
        print("Seqlet discovery completed.")

    def run_mutagenesis_analysis(self):
        """Calculates and plots In-silico Saturation Mutagenesis (ISM) and DeepLiftShap."""
        print("\n" + "="*50)
        print("--- 5. Mutagenesis Analysis (ISM vs DeepLiftShap) ---")

        # In-silico Saturation Mutagenesis (ISM) calculation
        def saturation_mutagenesis_effect_custom(model, X_input, device):
            # ISM only supports a single input sequence
            N, C, L = X_input.shape
            y_orig = model(X_input).detach().cpu().numpy().squeeze()
            X_ism = X_input.clone().repeat(C * L, 1, 1)
            base_indices = torch.arange(C).repeat(L)
            pos_indices = torch.arange(L).repeat_interleave(C)
            X_ism[torch.arange(C * L), :, pos_indices] = 0.0
            X_ism[torch.arange(C * L), base_indices, pos_indices] = 1.0
            y_mut = tpred(model, X_ism, device=str(device))
            diff_attr = y_mut.detach().cpu().numpy().squeeze() - y_orig  # type: ignore[union-attr]
            diff_attr = diff_attr.reshape(L, C).T
            return torch.tensor(diff_attr, dtype=torch.float32).unsqueeze(0)
    

        consensus_seq = self.data_prep.get_consensus_sequence()
        X_mut_test = one_hot_encode(consensus_seq).unsqueeze(0).float().to(self.device)

        # 5a. ISM Plot
        X_ism_attr = saturation_mutagenesis(self.model, X_mut_test.cpu(), batch_size=128, device=str(self.device))
        plt.figure(figsize=(12, 3))
        plot_logo(X_ism_attr[0].detach().cpu().numpy().astype(float), ax=plt.subplot(111))  # type: ignore[union-attr]
        plt.title("ISM (Prediction Difference)")
        save_or_show_plot("ISM_Attribution.png", self.config.output_dir)

        # 5b. DeepLiftShap Plot for the same sequence
        X_dl_attr = self.attr_core.run_deep_lift_shap(X_mut_test, self.config.n_shuffles, verbose=False)
        plt.figure(figsize=(12, 3))
        plot_logo(X_dl_attr[0].detach().cpu().numpy().astype(float), ax=plt.subplot(111))
        plt.title("DeepLiftShap Attribution on Consensus Sequence")
        save_or_show_plot("DeepLiftShap_Consensus.png", self.config.output_dir, subfolder="deeplift")

    def run_rc_comparison(self):
        """Compares forward and reverse complement predictions."""
        print("\n" + "="*50)
        print("--- 6. Forward vs. Reverse Complement Prediction ---")
        
        # use maria seqs
        X_orig = self.data_prep.load_maria_sequences().float().to(self.device)

        if X_orig.shape[0] == 0:
            print("Skipping RC comparison (no sequences available).")
            return
            
        X_rc = torch.flip(X_orig, [2])
        # Complement A->T, C->G, G->C, T->A (Index: 0->3, 1->2, 2->1, 3->0)
        X_rc = torch.index_select(X_rc, 1, torch.tensor([3, 2, 1, 0], dtype=torch.long, device=self.device))

        y = tpred(self.model, X_orig, device=str(self.device))
        y_rc = tpred(self.model, X_rc, device=str(self.device))

        plt.figure(figsize=(12, 3))
        plt.scatter(y.detach().cpu().numpy(), y_rc.detach().cpu().numpy(), alpha=0.5)  # type: ignore[union-attr]
        plt.xlabel("Forward Prediction")
        plt.ylabel("Reverse Complement Prediction")
        plt.title("Forward vs Reverse Complement Prediction")
        plt.plot([-1, 4], [-1, 4], color='red', linestyle='--')
        save_or_show_plot("Forward_vs_ReverseComplement_Prediction.png", self.config.output_dir)

    def run_epistasis_analysis(self):
        """Performs Pairwise SNV Epistasis Analysis."""
        print("\n" + "="*50)
        print("--- 7a. Pairwise SNV Epistasis Analysis (Additive Model) ---")

        X_consensus = one_hot_encode(self.data_prep.get_consensus_sequence()).unsqueeze(0).float().to(self.device)
        seq_len = X_consensus.shape[-1]
        y_orig = self.model(X_consensus).detach().cpu().numpy().squeeze()

        plot_x_dd, plot_y_dd, plot_c_dd , all_seqs = [], [], [], []
        offsets = np.linspace(-0.3, 0.3, 4)  # Offsets for plotting

        for j, i in itertools.combinations(range(seq_len), 2):
            orig_base_i = torch.argmax(X_consensus[0, :, i]).item()
            orig_base_j = torch.argmax(X_consensus[0, :, j]).item()
            
            for b1 in range(4): # Base at position i
                for b2 in range(4): # Base at position j
                    # Double mutant prediction
                    X_mut_ij = X_consensus.clone()
                    X_mut_ij[0, :, i], X_mut_ij[0, b1, i] = 0, 1
                    X_mut_ij[0, :, j], X_mut_ij[0, b2, j] = 0, 1
                    y_mut_ij = self.model(X_mut_ij).detach().cpu().numpy().squeeze()

                    # Single mutant predictions
                    X_mut_i, X_mut_j = X_consensus.clone(), X_consensus.clone()
                    X_mut_i[0, :, i], X_mut_i[0, b1, i] = 0, 1
                    X_mut_j[0, :, j], X_mut_j[0, b2, j] = 0, 1
                    y_mut_i = self.model(X_mut_i).detach().cpu().numpy().squeeze()
                    y_mut_j = self.model(X_mut_j).detach().cpu().numpy().squeeze()

                    # Epistasis score: ΔΔ = (y_mut_ij - y_mut_i - y_mut_j + y_orig)
                    dd_score = y_mut_ij - y_mut_i - y_mut_j + y_orig

                    plot_x_dd.append(i + offsets[b1])
                    plot_y_dd.append(j - offsets[b2])
                    plot_c_dd.append(0.0 if b1 == orig_base_i and b2 == orig_base_j else dd_score)
                    #store the actual mutated sequences for potential future use (e.g., tooltips in interactive plots) it needs to be the original sequence with the two mutations applied
                    all_seqs.append((i, b1, j, b2))  # Store positions and mutated bases as indices for now


                    


        import csv
        dict_base = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

        if self.config.output_dir:
            consensus_seq = list(self.data_prep.get_consensus_sequence())
            seq_len = len(consensus_seq)

            # --- Single mutations CSV ---
            single_csv = os.path.join(self.config.output_dir, "pairwise_snv_single_mutations.csv")
            with open(single_csv, mode='w', newline='') as sf:
                writer = csv.writer(sf)
                writer.writerow(["Mutated_Sequence", "Prediction"])
                for i in range(seq_len):
                    orig_base = consensus_seq[i]
                    for b in range(4):
                        mutated_base = dict_base[b]
                        if mutated_base == orig_base:
                            continue
                        mutated_seq = consensus_seq.copy()
                        mutated_seq[i] = mutated_base
                        mutated_seq_str = "".join(mutated_seq)

                        X_mut = one_hot_encode(mutated_seq_str).unsqueeze(0).float().to(self.device)
                        pred = self.model(X_mut).detach().cpu().numpy().tolist()
                        writer.writerow([mutated_seq_str, pred])

            # --- Double mutations CSV ---
            double_csv = os.path.join(self.config.output_dir, "pairwise_snv_double_mutations.csv")
            with open(double_csv, mode='w', newline='') as df:
                writer = csv.writer(df)
                writer.writerow(["Mutated_Sequence", "Prediction"])
                for (i, b1, j, b2) in all_seqs:
                    # skip if both mutated bases equal the original bases (i.e., no mutation)
                    if dict_base[b1] == consensus_seq[i] and dict_base[b2] == consensus_seq[j]:
                        continue
                    mutated_seq = consensus_seq.copy()
                    mutated_seq[i] = dict_base[b1]
                    mutated_seq[j] = dict_base[b2]
                    mutated_seq_str = "".join(mutated_seq)

                    X_mut = one_hot_encode(mutated_seq_str).unsqueeze(0).float().to(self.device)
                    pred = self.model(X_mut).detach().cpu().numpy().tolist()
                    writer.writerow([mutated_seq_str, pred])

            print(f"Single mutations saved to: {single_csv}")
            print(f"Double mutations saved to: {double_csv}")



        plt.figure(figsize=(12, 10))
        norm = TwoSlopeNorm(vcenter=0, vmin=np.min(plot_c_dd), vmax=np.max(plot_c_dd))
        sc = plt.scatter(plot_x_dd, plot_y_dd, c=plot_c_dd, cmap='coolwarm', norm=norm, alpha=0.7, s=1)
        plt.colorbar(sc, label='ΔΔ (Epistasis Score)')
        plt.title('Pairwise SNV Epistasis Scores (Additive Model)')
        save_or_show_plot("Pairwise_SNV_Epistasis_Additive.png", self.config.output_dir)


        
        # --- Pairwise SNV Epistasis Analysis (Additive Model) zoomed---
        print("--- 7b. Pairwise SNV Epistasis Analysis (Additive Model) Zoomed ---")

        # Zoom in on a region of interest, e.g., positions 30 to 44
        zoom_start, zoom_end = (seq_len // 2) - 6, (seq_len // 2) + 7

        # Convert to numpy arrays for masking
        plot_x_np = np.array(plot_x_dd)
        plot_y_np = np.array(plot_y_dd)
        plot_c_np = np.array(plot_c_dd)

        mask = (
            (plot_x_np >= zoom_start - 0.5) & (plot_x_np <= zoom_end + 0.5) &
            (plot_y_np >= zoom_start - 0.5) & (plot_y_np <= zoom_end + 0.5)
        )

        plt.figure(figsize=(10, 8))
        vmax = np.max(plot_c_np)
        vmin = np.min(plot_c_np)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

        # Plot the epistasis scores for the masked region
        sc = plt.scatter(
            plot_x_np[mask], plot_y_np[mask], c=plot_c_np[mask],
            cmap='bwr', alpha=1, s=20, edgecolor='k', linewidth=0.2, norm=norm
        )

        # Add the original consensus base to the diagonal
        for pos in range(zoom_start, zoom_end + 1):
            base = self.data_prep.get_consensus_sequence()[pos]
            plt.text(pos, pos, base, color='black', fontsize=12, ha='center', va='center', fontweight='bold')


        bases = ['A', 'C', 'G', 'T']
        legend_x0, legend_y0 = zoom_start - 0.5, zoom_end + 0.5  # top left corner
        cell_size = 0.6  # adjust as needed

        # Draw the legend grid with circles
        for i, base_x in enumerate(bases):
            for j, base_y in enumerate(bases):
                x = legend_x0 + i * cell_size
                y = legend_y0 - j * cell_size
                # Draw a circle instead of a rectangle
                circ = Circle((x + cell_size / 2, y - cell_size / 2), cell_size / 2.2, fill=False, edgecolor='k', linewidth=0.7)
                plt.gca().add_patch(circ)
                plt.text(x + cell_size / 2, y - cell_size / 2, f"{base_y}/{base_x}", 
                        ha='center', va='center', fontsize=8)

        # Axis labels for the legend
        for i, base in enumerate(bases):
            # x-axis (bottom of legend)
            plt.text(legend_x0 + i * cell_size + cell_size / 2, legend_y0 - 4 * cell_size - 0.1, base, 
                    ha='center', va='top', fontsize=9, fontweight='bold')
            # y-axis (left of legend)
            plt.text(legend_x0 - 0.1, legend_y0 - i * cell_size - cell_size / 2, base, 
                    ha='right', va='center', fontsize=9, fontweight='bold')


        # add red box around the middle four bases at the diagonal
        # the rectangle should be centered around the middle of the zoomed region and also diagonal
        middle = (zoom_start + zoom_end) // 2 - 1
        rect = Rectangle((middle, middle-0.5), 5, 0.8, linewidth=2, edgecolor='red', facecolor='none', linestyle='--', alpha=0.5)
        # tilt the rectangle to match the diagonal
        rect.set_angle(45)  # Rotate the rectangle by 45 degrees
        plt.gca().add_patch(rect)

        plt.colorbar(sc, label='Epistasis Score (additive, ΔΔ)')
        plt.xlabel('Position B')
        plt.ylabel('Position A')
        plt.title(f'Pairwise SNV Epistasis (Zoomed)')
        plt.xlim(zoom_start - 1, zoom_end + 1)
        plt.ylim(zoom_start - 1, zoom_end + 1)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        save_or_show_plot("Pairwise_SNV_Epistasis_Additive_Zoomed.png", self.config.output_dir)


    def run_epistasis_3D_analysis(self, start_idx=None, end_idx=None, plot_threshold=0.01):
        """
        Performs High-Order (3D) SNV Epistasis Analysis.
        
        Calculates the 3rd order interaction term:
        Delta^3 = y_ijk - (y_ij + y_ik + y_jk) + (y_i + y_j + y_k) - y_orig
        
        Args:
            start_idx (int): Start position for analysis (default: middle - 4)
            end_idx (int): End position for analysis (default: middle + 4)
            plot_threshold (float): Only analyze triplets with max interaction > threshold.
        """
        print("\n" + "="*50)
        print("--- 8. High-Order (3D) SNV Epistasis Analysis ---")

        # 1. Setup Data
        X_consensus = one_hot_encode(self.data_prep.get_consensus_sequence()).unsqueeze(0).float().to(self.device)
        seq_len = X_consensus.shape[-1]
        y_orig = self.model(X_consensus).detach().cpu().numpy().squeeze()

        # 2. Define Region of Interest
        if start_idx is None:
            start_idx = max(0, (seq_len // 2) - 4)
        if end_idx is None:
            end_idx = min(seq_len, (seq_len // 2) + 5)
            
        print(f"Analyzing region: {start_idx} to {end_idx} (Sequence Length: {seq_len})")
        region_range = list(range(start_idx, end_idx))
        
        # Store the MAX interaction score for each positional triplet (i, j, k)
        # We aggregate mutation combos to reduce noise: (i,j,k) -> max_score
        triplet_interactions = {} 

        # 3. Iterate through combinations of 3 positions
        total_combos = sum(1 for _ in itertools.combinations(region_range, 3))
        print(f"Calculating {total_combos} positional triplets...")

        count = 0
        for i, j, k in itertools.combinations(region_range, 3):
            count += 1
            if count % 10 == 0:
                print(f"Processing triplet {count}/{total_combos}...", end='\r')

            orig_base_i = torch.argmax(X_consensus[0, :, i]).item()
            orig_base_j = torch.argmax(X_consensus[0, :, j]).item()
            orig_base_k = torch.argmax(X_consensus[0, :, k]).item()
            
            # Track max interaction for this specific triplet of positions
            max_triplet_score = 0.0

            for b1 in range(4): # Base at i
                for b2 in range(4): # Base at j
                    for b3 in range(4): # Base at k
                        
                        # Skip if any base is wildtype (no epistasis possible)
                        if (b1 == orig_base_i) or (b2 == orig_base_j) or (b3 == orig_base_k):
                            continue

                        # --- Prepare Mutants ---
                        # 1. Triple Mutant (ijk)
                        X_ijk = X_consensus.clone()
                        X_ijk[0, :, i], X_ijk[0, b1, i] = 0, 1
                        X_ijk[0, :, j], X_ijk[0, b2, j] = 0, 1
                        X_ijk[0, :, k], X_ijk[0, b3, k] = 0, 1
                        y_ijk = self.model(X_ijk).detach().cpu().numpy().squeeze()

                        # 2. Double Mutants
                        X_ij = X_consensus.clone()
                        X_ij[0, :, i], X_ij[0, b1, i] = 0, 1
                        X_ij[0, :, j], X_ij[0, b2, j] = 0, 1
                        y_ij = self.model(X_ij).detach().cpu().numpy().squeeze()

                        X_ik = X_consensus.clone()
                        X_ik[0, :, i], X_ik[0, b1, i] = 0, 1
                        X_ik[0, :, k], X_ik[0, b3, k] = 0, 1
                        y_ik = self.model(X_ik).detach().cpu().numpy().squeeze()

                        X_jk = X_consensus.clone()
                        X_jk[0, :, j], X_jk[0, b2, j] = 0, 1
                        X_jk[0, :, k], X_jk[0, b3, k] = 0, 1
                        y_jk = self.model(X_jk).detach().cpu().numpy().squeeze()

                        # 3. Single Mutants
                        X_i = X_consensus.clone()
                        X_i[0, :, i], X_i[0, b1, i] = 0, 1
                        y_i = self.model(X_i).detach().cpu().numpy().squeeze()

                        X_j = X_consensus.clone()
                        X_j[0, :, j], X_j[0, b2, j] = 0, 1
                        y_j = self.model(X_j).detach().cpu().numpy().squeeze()

                        X_k = X_consensus.clone()
                        X_k[0, :, k], X_k[0, b3, k] = 0, 1
                        y_k = self.model(X_k).detach().cpu().numpy().squeeze()

                        # --- Calculate 3rd Order Epistasis ---
                        score = y_ijk - (y_ij + y_ik + y_jk) + (y_i + y_j + y_k) - y_orig
                        
                        # Keep the score with the largest magnitude for this triplet
                        if abs(score) > abs(max_triplet_score):
                            max_triplet_score = score
            
            # Store result if significant
            if abs(max_triplet_score) > plot_threshold:
                triplet_interactions[(i, j, k)] = max_triplet_score

        # 4. Visualization: "Sliced" Heatmaps
        if not triplet_interactions:
            print("\nNo significant 3rd order epistasis found above threshold.")
            return

        print("\nGenerating sliced heatmap visualization...")
        
        # Determine grid size for subplots
        num_positions = len(region_range)
        cols = int(np.ceil(np.sqrt(num_positions)))
        rows = int(np.ceil(num_positions / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows), constrained_layout=True)
        if num_positions == 1: axes = [axes] # Handle single subplot case
        axes = np.array(axes).reshape(-1) # Flatten

        # Determine Color Scale (Symmetric around 0)
        all_scores = list(triplet_interactions.values())
        max_val = max(abs(min(all_scores)), abs(max(all_scores))) if all_scores else 1.0
        norm = TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)

        # For each position in the range, create a slice
        im = None
        for idx, fixed_k in enumerate(region_range):
            ax = axes[idx]
            
            # Create a matrix for the slice (i vs j) given fixed_k
            # Initialize with NaNs so empty spots show as white/background
            slice_matrix = np.full((num_positions, num_positions), np.nan)
            
            # Fill the matrix
            # Check all triplets that contain fixed_k
            found_data = False
            for (p1, p2, p3), score in triplet_interactions.items():
                if fixed_k in (p1, p2, p3):
                    # Identify the other two positions
                    others = [p for p in (p1, p2, p3) if p != fixed_k]
                    
                    # If the triplet was like (30, 30, 31) - effectively 2D, skip or handle
                    if len(others) < 2: continue 
                    
                    # Map global positions to local matrix indices
                    # others[0] is x, others[1] is y (or vice versa, symmetric)
                    if others[0] in region_range and others[1] in region_range:
                        local_i = region_range.index(others[0])
                        local_j = region_range.index(others[1])
                        
                        slice_matrix[local_j, local_i] = score
                        slice_matrix[local_i, local_j] = score # Symmetric
                        found_data = True

            # Plot Heatmap
            # Use a gray background for the plot area to distinguish 0 from NaN
            ax.set_facecolor('#f0f0f0') 
            im = ax.imshow(slice_matrix, cmap='coolwarm', norm=norm, origin='lower')
            
            # Labels
            ax.set_title(f"Slice at Pos {fixed_k}", fontsize=10, fontweight='bold')
            
            # Ticks
            step = 1 if num_positions < 15 else 5
            tick_indices = np.arange(0, num_positions, step)
            tick_labels = [region_range[i] for i in tick_indices]
            
            ax.set_xticks(tick_indices)
            ax.set_xticklabels(tick_labels, fontsize=8, rotation=45)
            ax.set_yticks(tick_indices)
            ax.set_yticklabels(tick_labels, fontsize=8)

            if not found_data:
                ax.text(0.5, 0.5, "No Sig.\nInteractions", ha='center', va='center', fontsize=8, color='gray')

        # Hide empty subplots
        for i in range(num_positions, len(axes)):
            axes[i].axis('off')

        # Global Colorbar
        if im is None:
            print("No data to plot for 3rd-order epistasis slices.")
            plt.close(fig)
            return
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Max 3rd Order Epistasis (ΔΔΔ)', fontsize=12)

        plt.suptitle(f"3rd Order Epistasis Slices (Region {start_idx}-{end_idx})", fontsize=14)
        
        # Save
        save_or_show_plot("High_Order_SNV_Epistasis_Slices.png", self.config.output_dir)
        print("\n3D Analysis Complete (Visualized as Slices).")

    def run_attribution_method_comparison(self):
        """Compares DeepLiftShap and InputXGradient for a single sequence."""
        print("\n" + "="*50)
        print("--- 8. Attribution Method Comparison (DLSHAP vs. IXG) ---")

        X_orig = self.data_prep.load_maria_sequences()
        if X_orig.shape[0] < 1:
            print("Skipping comparison plot (no sequences available).")
            return
            
        X_comparison = X_orig[0:1].clone()
        seq_name = "First Maria Sequence"

        # 1. DeepLiftShap Calculation (now handled by AttributionCore)
        dl_attr = self.attr_core.run_deep_lift_shap(X_comparison, self.config.n_shuffles, verbose=False)
        dl_attr_np = dl_attr[0].detach().cpu().numpy().astype(float)

        # 2. InputXGradient Calculation (now handled by AttributionCore)
        ixg_attr = self.attr_core.calculate_input_x_gradient(X_comparison)
        ixg_attr_np = ixg_attr[0].detach().cpu().numpy().astype(float)
        
        # 3. Plotting
        plt.figure(figsize=(15, 6))
        ax1 = plt.subplot(211)
        plot_logo(dl_attr_np, ax=ax1)
        ax1.set_title(f"DeepLiftShap Attributions (N_shuffles={self.config.n_shuffles})")
        ax2 = plt.subplot(212)
        plot_logo(ixg_attr_np, ax=ax2)
        ax2.set_title("InputXGradient Attributions")
        plt.suptitle(f"Comparison of Attribution Methods for: {seq_name}", fontsize=14)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        save_or_show_plot(f"comparison_{seq_name}.png", self.config.output_dir)

    def run_conv_filter_visualization(self):
        """
        Visualizes convolutional filters by finding activating sequences,
        calculating a Position Frequency Matrix (PFM), converting it to
        an Information Content (bits) matrix, and plotting it as a sequence logo.
        This is adapted from the logic in pysster.Model.visualize_all_kernels.
        """
        print("\n" + "="*50)
        print("--- 9. Activation-based Filter Visualization (HTML Report) ---")

        # --- 1. Load Data ---
        try:
            peaks = pd.read_csv(self.config.peaks_file, sep="\t", usecols=(0, 1, 2), names=['chrom', 'start', 'end'])
            X_peaks = extract_loci(peaks, self.config.genome_fasta, in_window=self.config.peak_length).float()  # type: ignore[union-attr]
            X_peaks = X_peaks[X_peaks.sum(dim=(1, 2)) == self.config.peak_length].to(self.device)
            X_analysis = X_peaks[:min(500 if self.config.test else self.config.max_seqs, X_peaks.shape[0])]
            

            print(f"Loaded {X_analysis.shape[0]} sequences for analysis.")
        except Exception as e:
            print(f"Failed to load sequences: {e}. Skipping.")
            return

        # --- 2. Find all Conv1d and Linear layers to analyze ---
        layers = []
        name_to_module = {n: m for n, m in self.model.named_modules()}
        for name, module in name_to_module.items():
            if isinstance(module, torch.nn.Conv1d):
                layers.append((name, module, 'conv'))
            elif isinstance(module, torch.nn.Linear):
                layers.append((name, module, 'linear'))

        if not layers:
            print("No Conv1d (in_channels=4) or Linear layers found to visualize.")
            return

        # Helper: compute full receptive field (in input nucleotide positions)
        def compute_receptive_field_for_conv(layer_full_name: str, conv_module: torch.nn.Conv1d) -> int:
            """
            Estimate receptive field (in input positions) for the target conv by
            walking earlier convs in the same ModuleList (if present). Falls back
            to single-layer effective kernel: dilation*(k-1)+1.
            """
            # Try to detect ModuleList parent like 'conv_layers.2'
            m = re.match(r"(.+)\\.(\\d+)$", layer_full_name)
            if m:
                parent_name, idx_s = m.group(1), int(m.group(2))
                parent = name_to_module.get(parent_name)
                if parent is not None and isinstance(parent, (torch.nn.ModuleList, list)):
                    # iterate from 0..idx to compute receptive field
                    rf = 1
                    jump = 1
                    for j in range(0, idx_s + 1):
                        try:
                            layer = parent[j]
                        except Exception:
                            layer = None
                        if layer is None or not isinstance(layer, torch.nn.Conv1d):
                            continue
                        k = layer.kernel_size[0] if hasattr(layer, 'kernel_size') else 1
                        d = layer.dilation[0] if hasattr(layer, 'dilation') else 1
                        s = layer.stride[0] if hasattr(layer, 'stride') else 1
                        eff_k = d * (k - 1) + 1
                        rf = rf + (eff_k - 1) * jump
                        jump = jump * s
                    return int(rf)

            # Fallback: single conv effective kernel (in input space)
            k = conv_module.kernel_size[0] if hasattr(conv_module, 'kernel_size') else 1
            d = conv_module.dilation[0] if hasattr(conv_module, 'dilation') else 1
            eff_k = d * (k - 1) + 1
            return int(eff_k)

        # We'll process each layer separately and create one report per layer
        for layer_name, module, layer_type in layers:
            print(f"Visualizing filters for layer: {layer_name} (type={layer_type})")

            # Determine kernel length, out_channels and padding (conv) or emulate for linear
            padding = 0
            if layer_type == 'conv':
                kernel_len = module.kernel_size[0]
                out_channels = module.out_channels
                raw_padding = getattr(module, 'padding', 0)
                try:
                    if isinstance(raw_padding, (tuple, list)):
                        padding = int(raw_padding[0])
                    elif isinstance(raw_padding, int):
                        padding = int(raw_padding)
                    elif isinstance(raw_padding, str):
                        rp = raw_padding.lower()
                        if rp in ("valid", "none", "0"):
                            padding = 0
                        elif rp == "same":
                            padding = kernel_len // 2
                        else:
                            digits = "".join(ch for ch in rp if ch.isdigit())
                            padding = int(digits) if digits else 0
                    else:
                        padding = int(raw_padding)
                except Exception:
                    padding = 0
                if padding != 0:
                    print(f"  Layer {layer_name} uses padding={raw_padding} -> interpreted as {padding}.")

                # Compute activations via the conv layer by running the module
                # on the model's proper intermediate input. We use a forward
                # hook to ensure the module sees its true input (including
                # preceding layers) rather than trying to re-run it on raw
                # one-hot input when in_channels != 4.
                activations = None
                hook_handle = None
                try:
                    def _hook(m, inp, out):
                        nonlocal activations
                        # out is a Tensor; detach to CPU later when needed
                        activations = out.detach()

                    hook_handle = module.register_forward_hook(_hook)
                    # Run a single forward pass so the hook captures activations
                    _ = self.model(X_analysis)
                finally:
                    if hook_handle is not None:
                        hook_handle.remove()
                if activations is None:
                    # As a last resort, try running the conv directly on raw input
                    with torch.no_grad():
                        activations = module(X_analysis)
                activations = torch.relu(activations)

            else:  # linear
                # Try to interpret this Linear as a set of convolutional kernels
                w = module.weight.detach()
                out_channels, in_features = w.shape
                if in_features % 4 != 0:
                    print(f"  Skipping Linear layer {layer_name}: in_features={in_features} not divisible by 4.")
                    continue
                kernel_len = in_features // 4
                if kernel_len > X_analysis.shape[2]:
                    print(f"  Skipping Linear layer {layer_name}: inferred kernel_len={kernel_len} > sequence length {X_analysis.shape[2]}.")
                    continue

                # reshape weights to (out, 4, kernel_len) and run conv1d to get activations
                weight_reshaped = w.view(out_channels, 4, kernel_len).to(self.device)
                bias = module.bias.detach().to(self.device) if module.bias is not None else None
                with torch.no_grad():
                    activations = torch.nn.functional.conv1d(X_analysis, weight_reshaped, bias=bias)
                    activations = torch.relu(activations)

            # Now we have `activations` shaped (N, out_channels, L') that we can analyze similarly
            filter_results = []

            for i in range(activations.shape[1]):
                filter_activations = activations[:, i, :]
                if filter_activations.shape[1] == 0:
                    continue

                max_act_val, max_act_pos = torch.max(filter_activations, dim=1)

                # Thresholding
                try:
                    threshold = torch.quantile(max_act_val, 0.99)
                except Exception:
                    threshold = torch.max(max_act_val) if max_act_val.numel() > 0 else 0.0

                seq_indices = torch.where(max_act_val > threshold)[0]

                # Min sequence check
                if len(seq_indices) < 10:
                    min_seqs = min(max(10, activations.shape[1]), X_analysis.shape[0])
                    if max_act_val.shape[0] < min_seqs:
                        continue
                    _, top_indices = torch.topk(max_act_val, min_seqs)
                    seq_indices = top_indices

                if len(seq_indices) == 0:
                    continue

                importance_score = max_act_val[seq_indices].mean().item()
                num_activating_seqs = len(seq_indices)

                # Compute receptive field (in nucleotide positions) to extract
                # raw-input subsequences corresponding to activations.
                rf = compute_receptive_field_for_conv(layer_name, module) if layer_type == 'conv' else kernel_len

                motif_starts = max_act_pos[seq_indices]
                subsequences = []

                for k, seq_idx in enumerate(seq_indices):
                    # Map activation position to input window using receptive field
                    pos = int(motif_starts[k].item())
                    half = rf // 2
                    start = pos - half
                    end = start + rf
                    if start < 0 or end > X_analysis.shape[2]:
                        continue
                    subseq = X_analysis[seq_idx, :, start:end]
                    subsequences.append(subseq)

                if not subsequences:
                    continue

                subseq_stack = torch.stack(subsequences)
                pfm = subseq_stack.mean(dim=0)

                # --- PFM to Info Content ---
                pfm_np = pfm.detach().cpu().numpy().astype(float) + 1e-9
                pfm_np = pfm_np / pfm_np.sum(axis=0, keepdims=True)
                H = -np.sum(pfm_np * np.log2(pfm_np), axis=0)
                H_max = np.log2(4)
                I = H_max - H
                logo_matrix = pfm_np * I

                # --- Plotting to Memory Buffer ---
                fig = plt.figure(figsize=(max(6, kernel_len / 2), 2.5), dpi=100)
                ax = plt.subplot(111)
                plot_logo(logo_matrix, ax=ax)
                plt.title(f"{layer_name} - Filter {i} (n={num_activating_seqs})", fontsize=10)
                plt.ylabel("Bits")
                plt.tight_layout()

                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                plt.close(fig)
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')

                filter_results.append({
                    'id': i,
                    'importance': float(round(importance_score, 4)),
                    'count': num_activating_seqs,
                    'image': f"data:image/png;base64,{img_base64}"
                })

            # write out report for this layer (if any filters found)
            if filter_results:
                self.generate_html(filter_results, layer_name)


    def generate_html(self, results, layer_name):
        """
        Generates a standalone HTML file with embedded images and sorting logic.
        """
        # Sort by importance descending by default
        results.sort(key=lambda x: x['importance'], reverse=True)
        
        json_data = json.dumps(results)
        output_path = os.path.join(self.config.output_dir+"/Filter_Logos", f"Filter_Report_{layer_name}.html")
        
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CNN Filter Analysis: {layer_name}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .filter-card {{ transition: transform 0.2s; }}
        .filter-card:hover {{ transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1); }}
    </style>
</head>
<body class="bg-slate-50 text-slate-900 min-h-screen p-8">

    <div class="max-w-7xl mx-auto">
        <header class="mb-8 border-b pb-4 border-slate-200">
            <h1 class="text-3xl font-bold text-slate-800">Convolutional Filter Visualization</h1>
            <p class="text-slate-500 mt-2">Layer: <span class="font-mono text-blue-600">{layer_name}</span> | Total Filters: {len(results)}</p>
        </header>

        <div class="flex flex-col sm:flex-row justify-between items-center mb-6 gap-4">
            <div class="text-sm font-medium text-slate-600">
                Sorted by: <span id="current-sort" class="text-blue-600 font-bold">Importance</span>
            </div>
            <div class="flex gap-2 bg-white p-1 rounded-lg shadow-sm border border-slate-200">
                <button onclick="renderFilters('importance')" class="px-4 py-2 text-sm rounded hover:bg-slate-100 transition">Importance</button>
                <button onclick="renderFilters('count')" class="px-4 py-2 text-sm rounded hover:bg-slate-100 transition">Seq Count</button>
                <button onclick="renderFilters('id')" class="px-4 py-2 text-sm rounded hover:bg-slate-100 transition">Filter ID</button>
            </div>
        </div>

        <div id="grid-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            <!-- Cards will be injected here -->
        </div>
    </div>

    <script>
        const data = {json_data};

        function renderFilters(sortBy) {
            const container = document.getElementById('grid-container');
            container.innerHTML = '';
            
            document.getElementById('current-sort').innerText = 
                sortBy.charAt(0).toUpperCase() + sortBy.slice(1);

            // Sort data
            const sortedData = [...data].sort((a, b) => {{
                if (sortBy === 'id') return a.id - b.id;
                return b[sortBy] - a[sortBy];
            }});

            sortedData.forEach(item => {{
                const card = document.createElement('div');
                card.className = 'filter-card bg-white rounded-xl shadow border border-slate-200 overflow-hidden flex flex-col';
                
                card.innerHTML = `
                    <div class="p-4 border-b border-slate-100 flex justify-between items-center bg-slate-50">
                        <span class="font-mono font-bold text-slate-700">Filter #` + item.id + `</span>
                        <span class="text-xs font-bold px-2 py-1 rounded bg-blue-100 text-blue-700">
                            Score: ` + item.importance + `
                        </span>
                    </div>
                    <div class="p-4 bg-white flex justify-center items-center h-40">
                        <img src="` + item.image + `" alt="Filter Logo" class="max-h-full max-w-full object-contain cursor-pointer" onclick="openModal(this.src, ` + item.id + `, ` + item.importance + `, ` + item.count + `, '` + "{layer_name}" + `')">
                    </div>
                    <div class="px-4 py-3 bg-slate-50 border-t border-slate-100 text-xs text-slate-500 flex justify-between">
                        <span>` + item.count + ` sequences</span>
                        <span>Layer: {layer_name}</span>
                    </div>
                `;
                container.appendChild(card);
            }});
        }

                // Initial Render
                renderFilters('importance');

                // --- Modal / Lightbox for zooming images ---
                const modalHtml = `
                <div id="filter-modal" style="display:none;position:fixed;z-index:9999;left:0;top:0;width:100%;height:100%;background:rgba(0,0,0,0.75);align-items:center;justify-content:center;">
                    <div style="position:relative;max-width:95%;max-height:95%;margin:auto;padding:12px;background:#fff;border-radius:8px;box-shadow:0 10px 30px rgba(0,0,0,0.5);">
                        <button onclick="closeModal()" style="position:absolute;right:8px;top:8px;background:#111;color:#fff;border:none;padding:6px 10px;border-radius:4px;cursor:pointer;">Close</button>
                        <div style="text-align:center;">
                            <img id="modal-img" src="" style="max-width:100%;max-height:80vh;object-fit:contain;border-radius:4px;"/>
                            <div id="modal-caption" style="margin-top:8px;color:#333;font-size:14px;"></div>
                        </div>
                    </div>
                </div>`;

                document.body.insertAdjacentHTML('beforeend', modalHtml);

                function openModal(src, id, importance, count, layer) {
                        const modal = document.getElementById('filter-modal');
                        const img = document.getElementById('modal-img');
                        const cap = document.getElementById('modal-caption');
                        img.src = src;
                        cap.innerText = `Layer: ${layer} | Filter #${id} — Score: ${importance} — n=${count}`;
                        modal.style.display = 'flex';
                        modal.style.alignItems = 'center';
                        modal.style.justifyContent = 'center';
                }

                function closeModal() {
                        const modal = document.getElementById('filter-modal');
                        modal.style.display = 'none';
                }

                // Close modal on ESC
                document.addEventListener('keydown', function(e) {
                        if (e.key === 'Escape') closeModal();
                });
        </script>
</body>
</html>
        """

        # because the template contains many JavaScript/HTML curly braces. Convert
        # double-brace escapes back to single braces and then substitute the
        # intended placeholders safely.
        html_content = html_content.replace('{{', '{').replace('}}', '}')
        html_content = html_content.replace('{json_data}', json_data)
        html_content = html_content.replace('{layer_name}', layer_name)
        html_content = html_content.replace('{len(results)}', str(len(results)))

        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"Report generated successfully: {output_path}")
        except Exception as e:
            print(f"Failed to write HTML report: {e}")

# --- 5. Main Entry Point ---

def main():
    """Initializes configuration, loads the model, and runs the analysis pipeline."""
    try:
        config = AnalysisConfig()
        # initialize report collector on the class (save_or_show_plot uses it)
        AnalysisConfig._report_collector = None
        if config.output_dir and config.html_report:
            os.makedirs(config.output_dir, exist_ok=True)
            if 'all' in config.analyses or 'conv_filters' in config.analyses:
                os.makedirs(os.path.join(config.output_dir, "Filter_Logos"), exist_ok=True)
            AnalysisConfig._report_collector = ReportCollector(config.output_dir)

        model_loader = ModelLoader(config)
        model = model_loader.model
        device = model_loader.device

        interpreter = GenomicInterpreter(model, config, device)
        interpreter.run_all_analysis()

        print("\nAll analyses finished successfully.")

    except (RuntimeError, SystemExit) as e:
        # Handle exceptions gracefully if they originate from model loading or configuration
        if not isinstance(e, SystemExit):
            print(f"\nFATAL ERROR: {e}")
            sys.exit(1)
        # SystemExit (from argparse help or error) is allowed to pass

if __name__ == '__main__':
    main()
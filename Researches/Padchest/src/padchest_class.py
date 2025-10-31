"""
Filename: padchest_class.py
Description:
    PadChest class to perform Exploratory Data Analysis (EDA)
Author: Giovanni Officioso
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import ast
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


# ==============================================================================
# 1. DATASET LOADING AND BASIC STATISTICS
# ==============================================================================
class PadChestEDA:
    """
    Exploratory Data Analysis (EDA) for the PadChest dataset.
    """

    def __init__(self, data_path: Path, output_path: Path):
        """
        Initialize the PadChestEDA class with dataset and output paths.

        Args:
            data_path (Path): Path to the PadChest dataset.
            output_path (Path): Path to save EDA results.
        """
        self.data_path = data_path
        self.output_path = output_path

        # Load dataset
        print("Loading dataset...")
        if self.data_path.suffix == ".csv":
            self.dataset = pd.read_csv(self.data_path)
        elif self.data_path.suffix == ".json":
            self.dataset = pd.read_json(self.data_path)
        else:
            raise ValueError(
                f"Unsupported file format: {self.data_path.suffix}.\n"
                "Please use CSV or JSON."
            )

        print("Dataset loaded successfully.")
        print("-" * 50)
        print(f"Dataset contains {len(self.dataset)} records.")

        # Store analysis results
        self.stats = {}
        self.disease_labels = None
        self.label_matrix = None
        self.label_cols = None
        self.co_occurrence_matrix = None
        self.co_occur_df = None
        self.correlation_matrix = None

    def _identify_label_columns(self, label_prefix: str = "Label") -> List[str]:
        """
        Identify disease label columns in the dataset.

        Args:
            label_prefix (str): Prefix used to identify label columns.

        Returns:
            List of label column names.
        """
        # Method 1: Look for columns with specific prefix
        label_cols = [
            col for col in self.dataset.columns if col.startswith(label_prefix)
        ]

        if not label_cols:
            # Method 2: Look for binary columns (0/1 values only)
            print("No columns with prefix found. Looking for binary columns...")
            binary_mask = self.dataset.apply(
                lambda col: col.dropna().isin([0, 1, 0.0, 1.0]).all()
            )
            label_cols = self.dataset.columns[binary_mask].tolist()

        if not label_cols:
            print("WARNING: Could not automatically identify label columns.")
            print("Please specify label columns manually.")
            print("\nAvailable columns:")
            for col in self.dataset.columns:
                print(f"  - {col}")
            return []

        self.label_cols = label_cols
        print(f"\nIdentified {len(self.label_cols)} disease labels:")
        for label in self.label_cols[:10]:
            print(f"  - {label}")
        if len(label_cols) > 10:
            print(f"  ... and {len(label_cols) - 10} more")

        return label_cols

    def compute_basic_statistics(self) -> Dict:
        """
        Compute basic statistics of the dataset.

        Returns:
            Dictionary containing basic statistics.
        """
        print("\n" + "=" * 80)
        print("BASIC DATASET STATISTICS")
        print("=" * 80)

        stats = {
            "total_samples": len(self.dataset),
            "num_columns": len(self.dataset.columns),
            "column_names": list(self.dataset.columns),
        }

        # Check for image column
        img_cols = [col for col in self.dataset.columns if "image" in col.lower()]
        if img_cols:
            print(f"Image column identified: {img_cols[0]}")
            stats["img_column"] = img_cols[0]

        # Check for report column
        report_cols = [col for col in self.dataset.columns if "report" in col.lower()]
        if report_cols:
            text_col = report_cols[0]
            print(f"Report column identified: {text_col}")
            stats["report_column"] = text_col

            # Vectorized report analysis
            has_report = self.dataset[text_col].notna() & (
                self.dataset[text_col].astype(str).str.strip() != ""
            )
            stats["samples_with_reports"] = int(has_report.sum())
            stats["samples_without_reports"] = int((~has_report).sum())
            stats["report_coverage"] = float(has_report.mean() * 100)

            print("\nReport:")
            print(f"  Samples with reports: {stats['samples_with_reports']}")
            print(f"  Report coverage: {stats['report_coverage']:.2f}%")
            print(f"  Samples without reports: {stats['samples_without_reports']}")

        # Missing values analysis (vectorized)
        print("\nMissing Values:")
        missing = self.dataset.isnull().sum()
        missing_pct = (missing / len(self.dataset)) * 100

        missing_df = pd.DataFrame(
            {
                "Column": missing.index,
                "Missing_Count": missing.values,
                "Missing_Percentage": missing_pct.values,
            }
        )
        missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values(
            "Missing_Count", ascending=False
        )

        if not missing_df.empty:
            print(missing_df.head(10).to_string(index=False))
        else:
            print("  No missing values found!")

        stats["missing_values"] = missing_df.to_dict("records")
        self.stats["basic"] = stats

        return stats

    def _extract_disease_labels(
        self, label_columns: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract unique disease labels from the dataset.

        Args:
            label_columns (Optional[List[str]]): List of disease label columns.

        Returns:
            List of unique disease labels.
        """
        if label_columns is None:
            label_columns = self._identify_label_columns()

        if not label_columns:
            print("No label columns provided for disease label extraction.")
            return []

        print("\n" + "=" * 80)
        print("EXTRACTING DISEASE LABELS")
        print("=" * 80)

        # Optimized parsing function
        def _parse_cell(cell):
            if pd.isna(cell):
                return []

            if isinstance(cell, (list, tuple, set)):
                return [str(s).strip() for s in cell]

            if isinstance(cell, str):
                try:
                    val = ast.literal_eval(cell)
                    if isinstance(val, (list, tuple, set)):
                        return [str(s).strip() for s in val]
                    return [str(val).strip()]
                except (ValueError, SyntaxError):
                    parts = [
                        p.strip()
                        for p in cell.strip().strip("[]").split(",")
                        if p.strip()
                    ]
                    return parts

            return [str(cell)]

        # Vectorized label extraction
        if len(label_columns) == 1:
            list_of_labels = self.dataset[label_columns[0]].apply(_parse_cell).tolist()
        else:
            # Process all columns at once
            list_of_labels = []
            for idx in range(len(self.dataset)):
                combined = []
                for col in label_columns:
                    combined.extend(_parse_cell(self.dataset.at[idx, col]))

                # Remove duplicates preserving order
                seen = set()
                deduped = [x for x in combined if x and not (x in seen or seen.add(x))]
                list_of_labels.append(deduped)

        mlb = MultiLabelBinarizer(sparse_output=False)
        label_matrix_array = mlb.fit_transform(list_of_labels)

        self.label_matrix = pd.DataFrame(
            label_matrix_array,
            columns=mlb.classes_,
            index=self.dataset.index,
        )
        self.disease_labels = mlb.classes_.tolist()

        return self.disease_labels

    def _check_labelColumns_diseaseMatrix(
        self, label_columns: Optional[List[str]] = None
    ) -> None:
        """
        Check and extract the label matrix from the dataset.

        Args:
            label_columns (Optional[List[str]]): List of disease label columns.
        """
        if self.disease_labels is None or self.label_matrix is None:
            if label_columns:
                self.label_cols = label_columns
                self._extract_disease_labels(label_columns=label_columns)
            else:
                self._identify_label_columns()
                if self.label_cols:
                    self._extract_disease_labels(label_columns=self.label_cols)
                else:
                    raise RuntimeError(
                        "Could not identify label columns for disease matrix extraction."
                    )
            print("Label columns and disease matrix extracted.")
        else:
            print("Label columns and disease matrix already extracted.")

    def analyze_disease_distribution(self) -> Dict:
        """
        Analyze the distribution of diseases in the dataset.

        Returns:
            Dictionary containing disease distribution statistics.
        """
        self._check_labelColumns_diseaseMatrix()

        print("\n" + "=" * 80)
        print("DISEASE DISTRIBUTION")
        print("=" * 80)

        # Vectorized disease statistics computation
        positive_counts = self.label_matrix.sum(axis=0)
        n_samples = len(self.label_matrix)
        prevalence_pct = (positive_counts / n_samples) * 100

        disease_stats_df = pd.DataFrame(
            {
                "disease": self.disease_labels,
                "positive_count": positive_counts.values,
                "negative_count": n_samples - positive_counts.values,
                "prevalence_percentage": prevalence_pct.values,
            }
        ).sort_values("positive_count", ascending=False)

        print("\nDisease Prevalence (Top 10):")
        print(disease_stats_df.head(10).to_string(index=False))

        # Multi-label statistics (vectorized)
        labels_per_sample = self.label_matrix.sum(axis=1)

        print("\nMulti-label Statistics:")
        print(f"  Average labels per sample: {labels_per_sample.mean():.2f}")
        print(f"  Max labels in a single sample: {labels_per_sample.max():.0f}")
        print(f"  Min labels in a single sample: {labels_per_sample.min():.0f}")
        print(f"  Median labels per sample: {labels_per_sample.median():.2f}")
        print(f"  Samples with no labels: {(labels_per_sample == 0).sum()}")

        # Distribution of number of labels per sample
        print("\nDistribution of labels per sample:")
        label_counts = labels_per_sample.value_counts().sort_index()
        for num_labels, count in label_counts.items():
            print(
                f"  {int(num_labels)} labels: {count} samples "
                f"({count / n_samples * 100:.2f}%)"
            )

        stats = {
            "disease_statistics": disease_stats_df.to_dict("records"),
            "avg_labels_per_sample": float(labels_per_sample.mean()),
            "median_labels_per_sample": float(labels_per_sample.median()),
            "max_labels_in_sample": int(labels_per_sample.max()),
            "min_labels_in_sample": int(labels_per_sample.min()),
            "samples_with_no_labels": int((labels_per_sample == 0).sum()),
            "label_distribution": label_counts.to_dict(),
        }

        self.stats["distribution"] = stats
        return stats

    def compute_co_occurrence_matrix(
        self, label_columns: Optional[List[str]] = None, normalize: bool = True
    ) -> np.ndarray:
        """
        Compute the co-occurrence matrix of diseases.

        Args:
            label_columns (Optional[List[str]]): List of label column names
            normalize (bool): Whether to normalize by total samples

        Returns:
            Co-occurrence matrix (n_disease, n_disease)
        """
        self._check_labelColumns_diseaseMatrix(label_columns=label_columns)

        print("\n" + "=" * 80)
        print("DISEASE CO-OCCURRENCE MATRIX")
        print("=" * 80)

        # Compute co-occurrence matrix
        X = self.label_matrix.values
        co_occurrence = X.T @ X

        if normalize:
            co_occurrence = co_occurrence / len(X)
            print("Co-occurrence matrix normalized by total disease samples.")

        self.co_occurrence_matrix = co_occurrence

        print("Co-occurrence matrix computed successfully.")
        print(f"\nCo-occurrence matrix shape: {self.co_occurrence_matrix.shape}")

        # Diagonal statistics
        diag_vals = np.diag(self.co_occurrence_matrix)
        print(
            f"Diagonal (self-occurrence) range: [{diag_vals.min():.4f}, {diag_vals.max():.4f}]"
        )

        # Off-diagonal statistics
        n = self.co_occurrence_matrix.shape[0]
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag_vals = self.co_occurrence_matrix[off_diag_mask]
        print(
            f"Off-diagonal range: [{off_diag_vals.min():.4f}, {off_diag_vals.max():.4f}]"
        )

        # Find strongest co-occurrences
        print("\nStrongest Disease Co-occurrences (Top 10):")

        # Extract upper triangle indices
        i_idx, j_idx = np.triu_indices(n, k=1)

        # Convert labels to numpy array for indexing
        labels = np.array(self.disease_labels, dtype=object)

        co_occur_df = pd.DataFrame(
            {
                "disease_1": labels[i_idx],
                "disease_2": labels[j_idx],
                "co_occurrence": self.co_occurrence_matrix[i_idx, j_idx],
                "correlation": None,
            }
        ).sort_values("co_occurrence", ascending=False)

        self.co_occur_df = co_occur_df
        print(co_occur_df.head(10).to_string(index=False))

        return co_occurrence

    @staticmethod
    def _compute_phi_coefficient(label_matrix: pd.DataFrame) -> np.ndarray:
        """
        Compute phi coefficient for binary variables (vectorized).

        Phi coefficient measures association between two binary variables.
        """
        X = label_matrix.values
        n_samples = X.shape[0]

        # Contingency table components
        n11 = X.T @ X
        n1_ = X.sum(axis=0)
        n10 = n1_[:, None] - n11
        n01 = n1_ - n11
        n00 = n_samples - (n11 + n10 + n01)

        # Phi coefficient
        numerator = (n11 * n00) - (n10 * n01)
        denominator = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))

        phi_matrix = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator > 0,
        )

        return phi_matrix

    @staticmethod
    def _compute_jaccard_similarity(label_matrix: pd.DataFrame) -> np.ndarray:
        """
        Compute Jaccard similarity for binary variables (vectorized).

        Jaccard = |A ∩ B| / |A ∪ B|
        """
        X = label_matrix.values

        intersection = X.T @ X
        cardinality = X.sum(axis=0)
        union = cardinality[:, None] + cardinality - intersection

        jaccard_matrix = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union > 0,
        )

        return jaccard_matrix

    def compute_correlation_matrix(self, method: str = "phi") -> np.ndarray:
        """
        Compute the correlation matrix of diseases.

        Args:
            method (str): Correlation method ('phi' or 'jaccard')

        Returns:
            Correlation matrix (n_disease, n_disease)
        """
        if self.co_occurrence_matrix is None:
            print("Co-occurrence matrix not computed yet. Computing now...")
            self.compute_co_occurrence_matrix()

        print("\n" + "=" * 80)
        print(f"DISEASE CORRELATION MATRIX, METHOD: {method.upper()}")
        print("=" * 80)

        if method == "phi":
            self.correlation_matrix = self._compute_phi_coefficient(self.label_matrix)
            print("Correlation matrix computed using Phi coefficient.")
        elif method == "jaccard":
            self.correlation_matrix = self._compute_jaccard_similarity(
                self.label_matrix
            )
            print("Correlation matrix computed using Jaccard similarity.")
        else:
            raise ValueError(f"Unsupported correlation method: {method}")

        return self.correlation_matrix

    def analyze_conditional_probabilities(
        self, label_columns: Optional[List[str]] = None, top_k: int = 10
    ) -> Dict:
        """
        Analyze conditional probabilities between diseases.

        Args:
            label_columns (Optional[List[str]]): List of label column names
            top_k (int): Number of top conditional probabilities to return

        Returns:
            Dictionary containing conditional probability statistics.
        """
        self._check_labelColumns_diseaseMatrix(label_columns=label_columns)

        print("\n" + "=" * 80)
        print("DISEASE CONDITIONAL PROBABILITIES P(j|i)")
        print("=" * 80)

        # Vectorized computation
        X = self.label_matrix.values
        n_diseases = X.shape[1]

        # P(i) for each disease
        p_i = X.mean(axis=0)

        # P(i,j) from co-occurrence matrix (already computed or compute now)
        if self.co_occurrence_matrix is None:
            self.compute_co_occurrence_matrix(normalize=True)

        p_ij = np.asarray(self.co_occurrence_matrix, dtype=float)

        # P(j|i) = P(i,j) / P(i)
        conditional_probs = np.divide(
            p_ij, p_i[:, None], out=np.zeros_like(p_ij), where=p_i[:, None] > 0
        )

        # Set diagonal to 1 where P(i) > 0
        diag_idx = np.arange(n_diseases)
        conditional_probs[diag_idx, diag_idx] = np.where(p_i > 0, 1.0, np.nan)

        # Extract non-diagonal indices
        i_idx, j_idx = np.where(
            (~np.isnan(conditional_probs))
            & (np.arange(n_diseases)[:, None] != np.arange(n_diseases))
        )

        labels = np.array(self.disease_labels, dtype=object)

        cond_prob_df = pd.DataFrame(
            {
                "disease_i": labels[i_idx],
                "disease_j": labels[j_idx],
                "P(j|i)": conditional_probs[i_idx, j_idx],
            }
        ).sort_values("P(j|i)", ascending=False)

        print(f"\nTop {top_k} Conditional Probabilities P(j|i):")
        print(cond_prob_df.head(top_k).to_string(index=False))

        self.stats["conditional_probabilities"] = conditional_probs

        return {
            "conditional_probabilities": conditional_probs,
            "top_k_df": cond_prob_df.head(top_k),
        }

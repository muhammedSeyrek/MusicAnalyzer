"""
Validation and Benchmarking Framework for Music Analysis
==========================================================

This module provides tools for validating the music analysis algorithm
and generating benchmark results for academic publication.

Usage:
    python validation.py --test-file music.mp3 --expected western --scale "D Minor"
"""

import json
import pandas as pd
from pathlib import Path
import time
from typing import Dict, List, Tuple


class ValidationFramework:
    """
    Framework for validating and benchmarking music analysis results
    """

    def __init__(self):
        self.results = []
        self.ground_truth = {}

    def load_ground_truth(self, filepath: str):
        """
        Load ground truth annotations from JSON file

        Format:
        {
            "filename.mp3": {
                "system": "Western",
                "scale": "D Minor",
                "tempo": 120,
                "notes": "Rock ballad"
            }
        }
        """
        with open(filepath, 'r') as f:
            self.ground_truth = json.load(f)

    def validate_single_file(self, filename: str, analysis_result: Dict) -> Dict:
        """
        Validate analysis result against ground truth

        Returns:
        - accuracy metrics
        - confusion matrix data
        - detailed comparison
        """
        if filename not in self.ground_truth:
            return {'error': 'No ground truth for this file'}

        truth = self.ground_truth[filename]
        result = analysis_result

        # System classification accuracy
        system_correct = (
            (truth['system'] == 'Western' and result['is_western']) or
            (truth['system'] == 'Eastern' and not result['is_western'])
        )

        # Scale/Makam accuracy
        if truth['system'] == 'Western':
            scale_correct = truth['scale'] == result['western_tonality']
        else:
            scale_correct = truth['scale'] == result['eastern_makam']

        validation_result = {
            'filename': filename,
            'system_correct': system_correct,
            'scale_correct': scale_correct,
            'confidence': result['confidence'],
            'expected_system': truth['system'],
            'detected_system': result['system'],
            'expected_scale': truth['scale'],
            'detected_scale': result['western_tonality'] if result['is_western'] else result['eastern_makam'],
            'microtonal_ratio': result.get('microtonal_ratio', 0),
            'notes': truth.get('notes', '')
        }

        self.results.append(validation_result)
        return validation_result

    def calculate_metrics(self) -> Dict:
        """
        Calculate overall validation metrics

        Returns:
        - Accuracy (system classification)
        - Precision, Recall, F1 (Western vs Eastern)
        - Scale accuracy
        - Confidence distribution
        """
        if not self.results:
            return {}

        df = pd.DataFrame(self.results)

        # System classification accuracy
        system_accuracy = df['system_correct'].mean()

        # Scale accuracy (among correctly classified systems)
        scale_accuracy = df['scale_correct'].mean()

        # Precision and Recall for Western classification
        western_tp = len(df[(df['expected_system'] == 'Western') & (df['detected_system'] == 'Western')])
        western_fp = len(df[(df['expected_system'] == 'Eastern') & (df['detected_system'] == 'Western')])
        western_fn = len(df[(df['expected_system'] == 'Western') & (df['detected_system'] == 'Eastern')])
        western_tn = len(df[(df['expected_system'] == 'Eastern') & (df['detected_system'] == 'Eastern')])

        precision = western_tp / (western_tp + western_fp) if (western_tp + western_fp) > 0 else 0
        recall = western_tp / (western_tp + western_fn) if (western_tp + western_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Confidence statistics
        avg_confidence = df['confidence'].mean()
        conf_std = df['confidence'].std()

        # Confusion matrix
        confusion_matrix = {
            'true_western_predicted_western': western_tp,
            'true_western_predicted_eastern': western_fn,
            'true_eastern_predicted_western': western_fp,
            'true_eastern_predicted_eastern': western_tn
        }

        return {
            'system_accuracy': float(system_accuracy),
            'scale_accuracy': float(scale_accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'avg_confidence': float(avg_confidence),
            'confidence_std': float(conf_std),
            'confusion_matrix': confusion_matrix,
            'total_samples': len(df)
        }

    def generate_report(self, output_file: str = 'validation_report.html'):
        """
        Generate comprehensive validation report in HTML format
        """
        metrics = self.calculate_metrics()
        df = pd.DataFrame(self.results)

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Music Analysis Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e7f3fe; padding: 15px; margin: 10px 0; border-left: 4px solid #2196F3; }}
                .correct {{ color: green; font-weight: bold; }}
                .incorrect {{ color: red; font-weight: bold; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
            </style>
        </head>
        <body>
            <h1>🎵 Music Analysis Validation Report</h1>
            <p>Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Overall Metrics</h2>
            <div class="metric">
                <strong>System Classification Accuracy:</strong> {metrics['system_accuracy']:.2%}<br>
                <strong>Scale Detection Accuracy:</strong> {metrics['scale_accuracy']:.2%}<br>
                <strong>Precision (Western):</strong> {metrics['precision']:.2%}<br>
                <strong>Recall (Western):</strong> {metrics['recall']:.2%}<br>
                <strong>F1 Score:</strong> {metrics['f1_score']:.2%}<br>
                <strong>Average Confidence:</strong> {metrics['avg_confidence']:.2%} ± {metrics['confidence_std']:.2%}<br>
                <strong>Total Samples:</strong> {metrics['total_samples']}
            </div>

            <h2>Confusion Matrix</h2>
            <table>
                <tr>
                    <th></th>
                    <th>Predicted Western</th>
                    <th>Predicted Eastern</th>
                </tr>
                <tr>
                    <th>Actual Western</th>
                    <td class="correct">{metrics['confusion_matrix']['true_western_predicted_western']}</td>
                    <td class="incorrect">{metrics['confusion_matrix']['true_western_predicted_eastern']}</td>
                </tr>
                <tr>
                    <th>Actual Eastern</th>
                    <td class="incorrect">{metrics['confusion_matrix']['true_eastern_predicted_western']}</td>
                    <td class="correct">{metrics['confusion_matrix']['true_eastern_predicted_eastern']}</td>
                </tr>
            </table>

            <h2>Detailed Results</h2>
            {df.to_html(index=False, classes='dataframe')}
        </body>
        </html>
        """

        with open(output_file, 'w') as f:
            f.write(html)

        print(f"✅ Report generated: {output_file}")
        return output_file

    def export_results(self, output_file: str = 'validation_results.json'):
        """Export results to JSON for further analysis"""
        data = {
            'metrics': self.calculate_metrics(),
            'results': self.results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Results exported: {output_file}")


def create_test_dataset():
    """
    Create a sample test dataset for validation

    Returns:
    - ground_truth.json file with known music samples
    """
    test_data = {
        # Western Rock/Pop
        "GunsNRosesNovemberRain.mp3": {
            "system": "Western",
            "scale": "D Minor",
            "tempo": 82,
            "notes": "Rock ballad, heavy electric guitar"
        },
        "QueenBohemianRhapsody.mp3": {
            "system": "Western",
            "scale": "B Flat Major",
            "tempo": 72,
            "notes": "Complex arrangement, multiple sections"
        },
        "BeethovenMoonlightSonata.mp3": {
            "system": "Western",
            "scale": "C# Minor",
            "tempo": 54,
            "notes": "Classical piano, arpeggiated"
        },

        # Turkish Classical Music
        "HicazPesrev.mp3": {
            "system": "Eastern",
            "scale": "Hicaz",
            "tempo": 60,
            "notes": "Traditional Turkish classical, ney and kanun"
        },
        "RastSazsemaisi.mp3": {
            "system": "Eastern",
            "scale": "Rast",
            "tempo": 80,
            "notes": "Traditional instrumental, clear makam structure"
        },
        "NihavwendLonga.mp3": {
            "system": "Eastern",
            "scale": "Nihavend",
            "tempo": 120,
            "notes": "Upbeat, rhythmic"
        },
        "UşşakTaksim.mp3": {
            "system": "Eastern",
            "scale": "Uşşak",
            "tempo": 0,
            "notes": "Improvised taksim, microtonal bends"
        }
    }

    with open('ground_truth.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    print("✅ Test dataset created: ground_truth.json")
    print(f"   {len(test_data)} sample files defined")

    return test_data


if __name__ == "__main__":
    # Create sample test dataset
    create_test_dataset()

    print("\n📋 Validation Framework Ready")
    print("Usage:")
    print("  1. Add your music files to the project")
    print("  2. Run analysis on each file")
    print("  3. Use ValidationFramework to compare results")
    print("\nExample:")
    print("  validator = ValidationFramework()")
    print("  validator.load_ground_truth('ground_truth.json')")
    print("  validator.validate_single_file('music.mp3', analysis_result)")
    print("  validator.generate_report()")

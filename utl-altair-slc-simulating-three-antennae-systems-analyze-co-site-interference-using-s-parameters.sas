%let pgm=utl-altair-slc-simulating-three-antennae-systems-analyze-co-site-interference-using-s-parameters;

%stop_submission;

Altair slc simulating three antennae systems analyze co site interference using s parameters

Graphical Frequency Analysis
https://github.com/rogerjdeangelis/utl-altair-slc-simulating-three-antennae-systems-analyze-co-site-interference-using-s-parameters/blob/main/cosite_interference_analysis.png

Cosite interference analysis evaluates mutual coupling between antennas on the same platform,
using S-parameters to quantify signal leakage (e.g., S21 between ports)
versus frequency, which directly impacts system performance like receiver desensitization.?

Out of my depth here, however this may provide a template for Cosite Interference Analysis or
using python for simulations.

Too long to post in listserf, see github

github
https://github.com/rogerjdeangelis/utl-altair-slc-simulating-three-antennae-systems-analyze-co-site-interference-using-s-parameters

community.altair.com
https://community.altair.com/discussion/65289/cosite-interference-analysis-in-simlab-2026-0#latest

CONTENTS
    1 Folder structure and outputs
    2 Antennae diagrams
    3 SLC Inputs (initial conditions for all three antennae.
    4 SLC Python process
    5 SOLC Python outputs
       a frequency plots
       b summary reports
       c r transport files from panda frequency dataframes
       d create csv files from panda frequency dataframes (sane data as r rds transport files.
    6 SLC R process to convert r transport files to SLC tables
    7 R outputs (data behind frequency plots)
       a isolation.sas7bdat
       b reflection.sas7bdat
       c total_coupling.sas7bdat
       d transmission.sas7bdat
    8 python log
    9 r log

/*    __       _     _             ___                 _               _
/ |  / _| ___ | | __| | ___ _ __  ( _ )     ___  _   _| |_ _ __  _   _| |_ ___
| | | |_ / _ \| |/ _` |/ _ \ `__| / _ \/\  / _ \| | | | __| `_ \| | | | __/ __|
| | |  _| (_) | | (_| |  __/ |   | (_>  < | (_) | |_| | |_| |_) | |_| | |_\__ \
|_| |_|  \___/|_|\__,_|\___|_|    \___/\/  \___/ \__,_|\__| .__/ \__,_|\__|___/
                                                          |_|
*/

Outputs ( x 'tree "d:/lgs" /F /A | clip';=> creates the tree below system command )
Zipped in repo

D:\LGS
  |   isolation.sas7bdat                      Final slc output tables one per frequency plot
  |   reflection.sas7bdat
  |   total_coupling.sas7bdat
  |   transmission.sas7bdat
  |
  +---csv
  |       isolation_data.csv                  Final slc output tables in csv format
  |       isolation_summary_data.csv
  |       reflection_data.csv
  |       s_matrix_data.csv
  |       total_coupling_data.csv
  |       transmission_data.csv
  |       vswr_summary_data.csv
  |
  +---png
  |       cosite_interference_analysis.png    Final plot
  |
  +---rds                                     Export R transport file for proc r to make slc tables
  |       isolation_data.rds
  |       isolation_summary_data.rds
  |       reflection_data.rds
  |       s_matrix_data.rds
  |       total_coupling_data.rds
  |       transmission_data.rds
  |       vswr_summary_data.rds
  |
  |
  +---txt
  \       lgs_summary.txt                     Summary Reports



/*---
Cosite interference analysis evaluates mutual coupling between antennas on the same platform,
using S-parameters to quantify signal leakage (e.g., S21 between ports)
versus frequency, which directly impacts system performance like receiver desensitization.?

/*___                _                                   _ _
|___ \    __ _ _ __ | |_ ___ _ __  _ __   __ _  ___   __| (_) __ _  __ _ _ __ __ _ _ __ ___  ___
  __) |  / _` | `_ \| __/ _ \ `_ \| `_ \ / _` |/ _ \ / _` | |/ _` |/ _` | `__/ _` | `_ ` _ \/ __|
 / __/  | (_| | | | | ||  __/ | | | | | | (_| |  __/| (_| | | (_| | (_| | | | (_| | | | | | \__ \
|_____|  \__,_|_| |_|\__\___|_| |_|_| |_|\__,_|\___| \__,_|_|\__,_|\__, |_|  \__,_|_| |_| |_|___/
                                                                   |___/
*/

       Three-Antenna System Configuration

                                +-----------------+
                                |    Platform     |
                                | (Vehicle/Base)  |
                                +--------+--------+
                                         |
              +--------------------------|--------------------------+
              |                          |                          |
         +----v----+               +-----v-----+               +----v----+
         |         |               |           |               |         |
         | Antenna |               | Antenna   |               | Antenna |
         |   1     |               |   2       |               |   3     |
         |         |               |           |               |         |
         +----+----+               +-----+-----+               +----+----+
              |                          |                          |
              | Port 1                   | Port 2                   | Port 3
              |                          |                          |
         +----v--------------------------v--------------------------v---+
         |                    S-Parameter Matrix                        |
         |                                                              |
         |                    [S11   S21   S31]                         |
         |                    [S12   S22   S32]                         |
         |                    [S13   S23   S33]                         |
         |                                                              |
         +--------------------------------------------------------------+



         Three-Antenna System Configuration

                  Port 1        Port 2        Port 3
          +-------+     +-------+     +-------+
          |       |     |       |     |       |
          |  Ant1 |<--->|  Ant2 |<--->|  Ant3 |
          |       |     |       |     |       |
          +-------+     +-------+     +-------+
             ^             ^             ^
             |             |             |
             | S11=-10dB   | S22=-12dB   | S33=-11dB
             |             |             |
             v             v             v
           Self-        Self-         Self-
         Reflection   Reflection    Reflection

         Coupling Paths:
           Ant1 <-> Ant2: S21/S12 = -20dB
           Ant1 <-> Ant3: S31/S13 = -15dB
           Ant2 <-> Ant3: S32/S23 = -25dB


         Physical Layout
                                   Antenna 3
                            (rear)
                              ?
                              |
                              | S31/S13 = -15dB
                              |
         Antenna 1 <-----> Antenna 2
          (left)       S21/S12    (right)
                         -20dB
                              |
                              | S32/S23 = -25dB
                              |
                              ?
                          Antenna 3

---*/

/*____       _        _                   _
|___ /   ___| | ___  (_)_ __  _ __  _   _| |_ ___
  |_ \  / __| |/ __| | | `_ \| `_ \| | | | __/ __|
 ___) | \__ \ | (__  | | | | | |_) | |_| | |_\__ \
|____/  |___/_|\___| |_|_| |_| .__/ \__,_|\__|___/
                             |_|
*/

/*--- Coupling coefficients between antennas (normalized 0-1) ---*/
%let coupling_matrix=[[1.0, 0.2, 0.15],[0.2, 1.0, 0.25],[0.15, 0.25, 1.0]];

/*---
coupling_matrix
[1.0, 0.2, 0.15],    # S11, S21, S31
[0.2, 1.0, 0.25],    # S12, S22, S32
[0.15, 0.25, 1.0]    # S13, S23, S33
---*/

%let center_freq=2.4e9;

%let bandwidth=500e6;
%let Q_factors = [30, 35, 32];

%let noise_floor = -90;

/*--- Antenna resonant frequencies (slightly offset for realism) ---*/

%let  center_freq1 =  0.98;
/*--- center_freq2 no offset ---*/
%let  center_freq3 =  1.02;

/*  _                 _   _
| || |    _ __  _   _| |_| |__   ___  _ __   _ __  _ __ ___   ___ ___  ___ ___
| || |_  | `_ \| | | | __| `_ \ / _ \| `_ \ | `_ \| `__/ _ \ / __/ _ \/ __/ __|
|__   _| | |_) | |_| | |_| | | | (_) | | | || |_) | | | (_) | (_|  __/\__ \__ \
   |_|   | .__/ \__, |\__|_| |_|\___/|_| |_|| .__/|_|  \___/ \___\___||___/___/
         |_|    |___/                       |_|
*/

libname lgs sas7bdat  "d:/lgs";

/*--- slc output tables                    ---*/


Proc delete data=
   lgs.isolation
   lgs.reflection
   lgs.total_coupling
   lgs.transmission
   ;
;run;quit;

/*--- CSVs BEDIND ALL FOUR FREQUENCY PLOTS ---*/

%utlfkil(d:/lgs/csv/isolation_data.csv);
%utlfkil(d:/lgs/csv/reflection_data.csv);
%utlfkil(d:/lgs/csv/total_coupling_data.csv);
%utlfkil(vd:/lgs/csv/transmission_data.csv);

/*--- R TRANSPOT TABLES CREATED BY PYTHON  ---*/

%utlfkil(d:/lgs/rds/isolation_data.rds);
%utlfkil(d:/lgs/rds/reflection_data.rds);
%utlfkil(d:/lgs/rds/total_coupling_data.rds);
%utlfkil(d:/lgs/rds/transmission_data.rds);

%utlfkil(d:/lgs/txt/lgs_summary.txt);

%utlfkil(d:/png/cosite_interference_analysis.png);

%utl_slc_pybeginx(
   return=NA      /*- NA=disabled / return clipboard as maco var   -*/
  ,resolve=Y      /*- NA=disabled / resolve macros in python       -*/
  ,in=NA          /*- NA=disabled / input sas dataset              -*/
  ,out=NA         /*- NA=disabled / output work.female             -*/
  ,py2r=NA );     /*- NA=disabled / temp/rds,rds                   -*/
cards4;
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import sys
from matplotlib.gridspec import GridSpec
from datetime import date
import pyreadr as pr

# Set random seed for reproducibility
np.random.seed(42)

class ThreeAntennaSystem:
    """Simulate a three-antenna system with mutual coupling"""

    def __init__(self, center_freq=&center_freq, bandwidth=&bandwidth):
        """
        Initialize the antenna system
        center_freq: Center frequency in Hz (default: 2.4 GHz)
        bandwidth: Bandwidth in Hz (default: 500 MHz)
        """
        self.center_freq = center_freq
        self.bandwidth = bandwidth

        # Antenna resonant frequencies (slightly offset for realism)
        self.f_res = [
            center_freq * &center_freq1,  # Antenna 1
            center_freq,         # Antenna 2
            center_freq * &center_freq3   # Antenna 3
        ]

        # Quality factors for each antenna
        self.Q_factors = &Q_factors

        # Coupling coefficients between antennas (normalized 0-1)
        #self.coupling_matrix = np.array([
        #    [1.0, 0.2, 0.15],    # S11, S21, S31
        #    [0.2, 1.0, 0.25],    # S12, S22, S32
        #    [0.15, 0.25, 1.0]    # S13, S23, S33
        #])

        self.coupling_matrix = np.array(&coupling_matrix)

        # Noise floor for realistic simulation
        self.noise_floor = &noise_floor  # dB

    def generate_s_parameters(self, freq_points=501):
        """Generate S-parameters across frequency range"""
        frequencies = np.linspace(
            self.center_freq - self.bandwidth/2,
            self.center_freq + self.bandwidth/2,
            freq_points
        )

        # Initialize S-parameter matrix: [freq, port_out, port_in]
        s_params = np.zeros((len(frequencies), 3, 3), dtype=complex)

        # Generate self-impedance (reflection coefficients)
        for i in range(3):
            for j, freq in enumerate(frequencies):
                # Resonant response for reflection coefficients
                omega = 2 * np.pi * freq
                omega_res = 2 * np.pi * self.f_res[i]

                # Second-order resonant response
                s_mag = 1 / np.sqrt(1 + (2 * self.Q_factors[i] *
                                        (freq/self.f_res[i] - self.f_res[i]/freq))**2)

                # Add phase shift
                phase = -2 * np.arctan(2 * self.Q_factors[i] *
                                      (freq/self.f_res[i] - 1))

                s_params[j, i, i] = s_mag * np.exp(1j * phase)

        # Generate mutual coupling (transmission coefficients)
        for out_port in range(3):
            for in_port in range(3):
                if out_port != in_port:
                    coupling = self.coupling_matrix[out_port, in_port]

                    # Frequency-dependent coupling (bandpass characteristic)
                    for j, freq in enumerate(frequencies):
                        # Coupling decreases away from center frequency
                        freq_factor = np.exp(-((freq - self.center_freq)**2) /
                                           (2 * (self.bandwidth/4)**2))

                        # Transmission coefficient with realistic phase
                        s_mag = coupling * freq_factor * 0.5
                        phase = -2 * np.pi * np.abs(out_port - in_port) * freq / 3e8 * 0.1

                        s_params[j, out_port, in_port] = s_mag * np.exp(1j * phase)

        # Convert to dB and add realistic noise
        s_params_db = 20 * np.log10(np.abs(s_params) + 1e-10)
        noise = np.random.normal(0, 0.5, s_params_db.shape)  # 0.5 dB noise
        s_params_db += noise
        s_params_db = np.maximum(s_params_db, self.noise_floor)

        return frequencies, s_params_db

    def create_frequency_dataframes(self, frequencies, s_params_db, results):
        """Create pandas DataFrames for frequency domain data"""

        freq_ghz = frequencies / 1e9

        # DataFrame 1: Reflection Coefficients
        reflection_df = pd.DataFrame({
            'Frequency_GHz': freq_ghz,
            'S11_dB': s_params_db[:, 0, 0],
            'S22_dB': s_params_db[:, 1, 1],
            'S33_dB': s_params_db[:, 2, 2]
        })

        # DataFrame 2: Transmission Coefficients (Coupling)
        transmission_df = pd.DataFrame({
            'Frequency_GHz': freq_ghz,
            'S21_dB': s_params_db[:, 1, 0],  # Port 1 to Port 2
            'S31_dB': s_params_db[:, 2, 0],  # Port 1 to Port 3
            'S32_dB': s_params_db[:, 2, 1]   # Port 2 to Port 3
        })

        # DataFrame 3: Isolation (negative of coupling)
        isolation_df = pd.DataFrame({
            'Frequency_GHz': freq_ghz,
            'Isolation_Ant1_2_dB': -s_params_db[:, 1, 0],  # Negative of S21
            'Isolation_Ant1_3_dB': -s_params_db[:, 2, 0],  # Negative of S31
            'Isolation_Ant2_3_dB': -s_params_db[:, 2, 1]   # Negative of S32
        })

        # DataFrame 4: Total Coupling to each antenna
        total_coupling_df = pd.DataFrame({
            'Frequency_GHz': freq_ghz,
            'Total_Coupling_Ant1_dB': results['total_coupling']['Antenna 1'],
            'Total_Coupling_Ant2_dB': results['total_coupling']['Antenna 2'],
            'Total_Coupling_Ant3_dB': results['total_coupling']['Antenna 3']
        })

        # DataFrame 5: VSWR Summary
        vswr_data = []
        for i in range(3):
            vswr_data.append({
                'Antenna': f'Antenna {i+1}',
                'VSWR_Min': results['vswr'][f'Antenna {i+1}']['min'],
                'VSWR_Max': results['vswr'][f'Antenna {i+1}']['max'],
                'VSWR_Center': results['vswr'][f'Antenna {i+1}']['center']
            })
        vswr_summary_df = pd.DataFrame(vswr_data)

        # DataFrame 6: Isolation Summary
        isolation_summary_data = []
        for iso_data in results['isolation']:
            isolation_summary_data.append({
                'Antenna_Pair': iso_data['pair'],
                'S_Parameter': iso_data['label'],
                'Min_Isolation_dB': iso_data['min_isolation'],
                'Avg_Isolation_dB': iso_data['avg_isolation'],
                'Worst_Freq_GHz': iso_data['worst_freq']
            })
        isolation_summary_df = pd.DataFrame(isolation_summary_data)

        # DataFrame 7: S-Parameter Matrix at Center Frequency
        center_idx = len(frequencies) // 2
        s_matrix_data = []
        for i in range(3):
            for j in range(3):
                s_matrix_data.append({
                    'Input_Port': f'Port {j+1}',
                    'Output_Port': f'Port {i+1}',
                    'S_Parameter': f'S{i+1}{j+1}',
                    'Magnitude_dB': s_params_db[center_idx, i, j],
                    'Frequency_GHz': freq_ghz[center_idx]
                })
        s_matrix_df = pd.DataFrame(s_matrix_data)

        return {
            'reflection': reflection_df,
            'transmission': transmission_df,
            'isolation': isolation_df,
            'total_coupling': total_coupling_df,
            'vswr_summary': vswr_summary_df,
            'isolation_summary': isolation_summary_df,
            's_matrix': s_matrix_df
        }

    def calculate_isolation(self, s_params_db):
        """Calculate isolation metrics from S-parameters"""
        # Minimum isolation (worst-case coupling)
        s21_min = np.min(s_params_db[:, 1, 0])
        s31_min = np.min(s_params_db[:, 2, 0])
        s32_min = np.min(s_params_db[:, 2, 1])

        # Average isolation in operating band
        center_idx = len(s_params_db) // 2
        band_idx = slice(center_idx - 50, center_idx + 50)

        avg_isolation = {
            'S21_avg': np.mean(s_params_db[band_idx, 1, 0]),
            'S31_avg': np.mean(s_params_db[band_idx, 2, 0]),
            'S32_avg': np.mean(s_params_db[band_idx, 2, 1])
        }

        return s21_min, s31_min, s32_min, avg_isolation

    def analyze_cosite_interference(self, frequencies, s_params_db):
        """Perform co-site interference analysis"""
        results = {}

        # 1. Reflection coefficient analysis (match quality)
        results['vswr'] = {}
        for i in range(3):
            s11 = s_params_db[:, i, i]
            # Calculate VSWR from S11
            gamma = 10**(s11/20)
            vswr = (1 + gamma) / (1 - gamma)
            results['vswr'][f'Antenna {i+1}'] = {
                'min': np.min(vswr),
                'max': np.max(vswr),
                'center': vswr[len(vswr)//2]
            }

        # 2. Isolation analysis
        isolation_data = []
        pairs = [(0, 1, 'S21'), (0, 2, 'S31'), (1, 2, 'S32')]

        for i, j, label in pairs:
            coupling = s_params_db[:, j, i]
            isolation = -coupling  # Isolation is negative of coupling in dB
            isolation_data.append({
                'pair': f'{i+1}-{j+1}',
                'label': label,
                'min_isolation': np.min(isolation),
                'avg_isolation': np.mean(isolation[50:-50]),
                'worst_freq': frequencies[np.argmin(isolation)] / 1e9
            })

        results['isolation'] = isolation_data

        # 3. Calculate total coupling to each antenna
        results['total_coupling'] = {}
        for i in range(3):
            total_power = 0
            for j in range(3):
                if i != j:
                    # Convert from dB to linear, sum power, convert back to dB
                    power_linear = 10**(s_params_db[:, j, i]/10)
                    total_power += power_linear
            results['total_coupling'][f'Antenna {i+1}'] = 10 * np.log10(total_power + 1e-10)

        return results

def plot_s_parameters(frequencies, s_params_db, results, dataframes=None):
    """Create comprehensive visualization of S-parameters"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig)

    # Convert frequencies to GHz for plotting
    freq_ghz = frequencies / 1e9

    # Plot 1: Reflection coefficients (S11, S22, S33)
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(3):
        ax1.plot(freq_ghz, s_params_db[:, i, i],
                label=f'S{i+1}{i+1}', color=colors[i], linewidth=2)

    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Reflection Coefficients')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([-30, 0])

    # Plot 2: Transmission coefficients (coupling)
    ax2 = fig.add_subplot(gs[0, 1])
    coupling_pairs = [(0, 1, 'S21'), (0, 2, 'S31'), (1, 2, 'S32')]
    for i, j, label in coupling_pairs:
        ax2.plot(freq_ghz, s_params_db[:, j, i],
                label=label, linewidth=2)

    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Transmission Coefficients (Coupling)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([-50, -10])

    # Plot 3: Isolation between antenna pairs
    ax3 = fig.add_subplot(gs[0, 2])
    for i, j, label in coupling_pairs:
        isolation = -s_params_db[:, j, i]  # Isolation is negative of coupling
        ax3.plot(freq_ghz, isolation,
                label=f'Ant {i+1}-{j+1}', linewidth=2)

    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Isolation (dB)')
    ax3.set_title('Antenna-to-Antenna Isolation')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim([10, 50])

    # Plot 4: 3D visualization of coupling matrix at center frequency
    ax4 = fig.add_subplot(gs[1, :])
    center_idx = len(frequencies) // 2

    # Create coupling matrix heatmap
    coupling_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            coupling_matrix[i, j] = s_params_db[center_idx, j, i]

    im = ax4.imshow(coupling_matrix, cmap='viridis', aspect='auto')
    ax4.set_title(f'S-parameter Matrix at {freq_ghz[center_idx]:.2f} GHz')
    ax4.set_xticks([0, 1, 2])
    ax4.set_yticks([0, 1, 2])
    ax4.set_xticklabels(['Port 1', 'Port 2', 'Port 3'])
    ax4.set_yticklabels(['Port 1', 'Port 2', 'Port 3'])

    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax4.text(j, i, f'{coupling_matrix[i, j]:.1f} dB',
                          ha="center", va="center",
                          color="white" if coupling_matrix[i, j] < -25 else "black")

    plt.colorbar(im, ax=ax4, label='Magnitude (dB)')

    # Plot 5: Total coupling to each antenna
    ax5 = fig.add_subplot(gs[2, 0])
    for i in range(3):
        total_coupling = results['total_coupling'][f'Antenna {i+1}']
        ax5.plot(freq_ghz, total_coupling,
                label=f'Antenna {i+1}', linewidth=2)

    ax5.set_xlabel('Frequency (GHz)')
    ax5.set_ylabel('Total Coupling (dB)')
    ax5.set_title('Total Interference Coupling per Antenna')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_ylim([-40, -10])

    # Plot 6: VSWR analysis
    ax6 = fig.add_subplot(gs[2, 1])
    antenna_labels = []
    vswr_center = []

    for i in range(3):
        antenna_labels.append(f'Ant {i+1}')
        vswr_center.append(results['vswr'][f'Antenna {i+1}']['center'])

    bars = ax6.bar(antenna_labels, vswr_center, color=colors)
    ax6.set_ylabel('VSWR at Center Frequency')
    ax6.set_title('Voltage Standing Wave Ratio')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add VSWR threshold lines
    ax6.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='VSWR=2.0')
    ax6.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='VSWR=1.5')
    ax6.legend()

    # Plot 7: Isolation summary
    ax7 = fig.add_subplot(gs[2, 2])
    isolation_pairs = []
    min_isolation_values = []

    for iso_data in results['isolation']:
        isolation_pairs.append(iso_data['pair'])
        min_isolation_values.append(iso_data['min_isolation'])

    bars = ax7.bar(isolation_pairs, min_isolation_values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax7.set_ylabel('Minimum Isolation (dB)')
    ax7.set_title('Worst-case Antenna Isolation')
    ax7.grid(True, alpha=0.3, axis='y')

    # Add isolation requirement line
    ax7.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20 dB Requirement')
    ax7.legend()

    # Add value labels on bars
    for bar, val in zip(bars, min_isolation_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} dB', ha='center', va='bottom')

    plt.suptitle('Three-Antenna Co-site Interference Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def generate_report(results, frequencies, s_params_db, dataframes):
    """Generate a text report of the analysis"""

    with open("d:/lgs/txt/lgs_summary.txt", 'a') as g:
        print("Starting Co-site Interference Analysis for Three-Antenna System...", file=g)
        print("="*70, file=g)
        print("CO-SITE INTERFERENCE ANALYSIS REPORT", file=g)
        print("="*70, file=g)

        print("\n1. REFLECTION COEFFICIENT SUMMARY (S11, S22, S33):", file=g)
        print("-"*50, file=g)
        for i in range(3):
            vswr_data = results['vswr'][f'Antenna {i+1}']
            print(f"Antenna {i+1} (S{i+1}{i+1}):", file=g)
            print(f"  - VSWR range: {vswr_data['min']:.2f} - {vswr_data['max']:.2f}", file=g)
            print(f"  - VSWR at center: {vswr_data['center']:.2f}", file=g)
            print(f"  - Match quality: {'Good' if vswr_data['center'] < 2.0 else 'Marginal' if vswr_data['center'] < 3.0 else 'Poor'}", file=g)

        print("\n2. ISOLATION ANALYSIS:", file=g)
        print("-"*50, file=g)
        for iso_data in results['isolation']:
            print(f"{iso_data['label']} (Antenna {iso_data['pair'][0]} to {iso_data['pair'][-1]}):", file=g)
            print(f"  - Minimum isolation: {iso_data['min_isolation']:.1f} dB", file=g)
            print(f"  - Average isolation: {iso_data['avg_isolation']:.1f} dB", file=g)
            print(f"  - Worst frequency: {iso_data['worst_freq']:.3f} GHz", file=g)
            print(f"  - Status: {'? PASS' if iso_data['min_isolation'] > 20 else '? MARGINAL' if iso_data['min_isolation'] > 15 else '? FAIL'}", file=g)

        print("\n3. DATAFRAME SUMMARIES:", file=g)
        print("-"*50, file=g)
        print("DataFrames created for frequency domain analysis:", file=g)
        for df_name, df in dataframes.items():
            print(f"  - {df_name}: {df.shape[0]} rows × {df.shape[1]} columns", file=g)

        print("\n4. INTERFERENCE ASSESSMENT:", file=g)
        print("-"*50, file=g)

        # Calculate worst-case scenarios
        center_idx = len(frequencies) // 2
        worst_coupling = np.min(s_params_db[:, :, :])
        worst_pair = np.unravel_index(np.argmin(s_params_db[:, :, :]), s_params_db.shape)

        print(f"Worst-case coupling: {worst_coupling:.1f} dB", file=g)
        print(f"Occurs at: {frequencies[worst_pair[0]]/1e9:.3f} GHz", file=g)
        print(f"Between: Port {worst_pair[2]+1} ? Port {worst_pair[1]+1}", file=g)

        print("\n5. RECOMMENDATIONS:", file=g)
        print("-"*50, file=g)

        # Generate recommendations based on analysis
        recommendations = []

        # Check isolation requirements
        isolation_status = all(iso['min_isolation'] > 20 for iso in results['isolation'])
        if not isolation_status:
            recommendations.append("- Consider increasing antenna spacing or adding shielding")
            recommendations.append("- Implement frequency separation or filtering")

        # Check VSWR
        vswr_status = all(v['center'] < 2.0 for v in results['vswr'].values())
        if not vswr_status:
            recommendations.append("- Optimize antenna matching networks")

        if not recommendations:
            recommendations.append("- System meets basic co-site requirements")
            recommendations.append("- Monitor performance with actual hardware")

        for rec in recommendations:
            print(rec, file=g)

        print("\n" + "="*70, file=g)

        # Save DataFrames to CSV files
        print("\nSaving DataFrames to CSV files...", file=g)
        for df_name, df in dataframes.items():
            filename = f"d:/lgs/csv/{df_name}_data.csv"
            df.to_csv(filename, index=False)
            print(f"  - Saved {df_name} to {filename}", file=g)

        print("\nSaving DataFrames to R rds files...", file=g)
        for df_name, df in dataframes.items():
            filename = f"d:/lgs/rds/{df_name}_data.rds"
            # Correct: Pass the DataFrame 'df', not the name '{df_name}'
            pr.write_rds(filename,df)  # Note: parameters swapped - DataFrame first, then filename
            print(f"  - Saved {df_name} to {filename}", file=g)

def main():
    """Main function to run the analysis"""

    with open("d:/lgs/txt/lgs_summary.txt", 'w') as h:

        # Create antenna system
        antenna_system = ThreeAntennaSystem(
            center_freq=2.4e9,  # 2.4 GHz center frequency
            bandwidth=500e6     # 500 MHz bandwidth
        )

        # Generate S-parameters
        frequencies, s_params_db = antenna_system.generate_s_parameters()

        # Perform analysis
        results = antenna_system.analyze_cosite_interference(frequencies, s_params_db)

        # Create pandas DataFrames for frequency domain data
        dataframes = antenna_system.create_frequency_dataframes(frequencies, s_params_db, results)

        # Display DataFrame information
        print("="*60, file=h)
        print("FREQUENCY DOMAIN DATAFRAMES CREATED:", file=h)
        print("="*60, file=h)

        for name, df in dataframes.items():
            print(f"\n{name.upper()} DataFrame:", file=h)
            print("-"*40, file=h)
            print(f"Shape: {df.shape}", file=h)
            print(f"Columns: {list(df.columns)}", file=h)
            if df.shape[0] > 5:
                print("\nFirst 5 rows:", file=h)
                print(df.head(), file=h)
            else:
                print(df, file=h)
            print("\nSummary statistics:", file=h)
            if 'Frequency_GHz' in df.columns and df.shape[0] > 3:
                print(df.describe(), file=h)

        # Generate report with DataFrames
        generate_report(results, frequencies, s_params_db, dataframes)

        # Create visualizations
        fig = plot_s_parameters(frequencies, s_params_db, results, dataframes)

        # Save results
        plt.savefig('d:/lgs/png/cosite_interference_analysis.png', dpi=150, bbox_inches='tight')
        print("\nAnalysis complete! Results saved to 'd:/lgs/png/cosite_interference_analysis.png'", file=h)

        # Display additional metrics
        print("\nAdditional Metrics:", file=h)
        print("-"*40, file=h)

        # Calculate and display isolation metrics
        s21_min, s31_min, s32_min, avg_isolation = antenna_system.calculate_isolation(s_params_db)

        print(f"Minimum S21 (Ant1?Ant2): {s21_min:.1f} dB", file=h)
        print(f"Minimum S31 (Ant1?Ant3): {s31_min:.1f} dB", file=h)
        print(f"Minimum S32 (Ant2?Ant3): {s32_min:.1f} dB", file=h)

        print(f"\nAverage Isolation in Operating Band:", file=h)
        print(f"S21 average: {avg_isolation['S21_avg']:.1f} dB", file=h)
        print(f"S31 average: {avg_isolation['S31_avg']:.1f} dB", file=h)
        print(f"S32 average: {avg_isolation['S32_avg']:.1f} dB", file=h)

        # Example: Access specific data from DataFrames
        print("\n" + "="*60, file=h)
        print("EXAMPLE DATA ACCESS FROM DATAFRAMES:", file=h)
        print("="*60, file=h)

        # Get center frequency data
        center_idx = len(frequencies) // 2
        center_freq_ghz = frequencies[center_idx] / 1e9

        print(f"\nData at center frequency ({center_freq_ghz:.2f} GHz):", file=h)
        print("-"*40, file=h)

        # Reflection coefficients at center frequency
        reflection_center = dataframes['reflection'].iloc[center_idx]
        print(f"Reflection Coefficients:", file=h)
        print(f"  S11: {reflection_center['S11_dB']:.1f} dB", file=h)
        print(f"  S22: {reflection_center['S22_dB']:.1f} dB", file=h)
        print(f"  S33: {reflection_center['S33_dB']:.1f} dB", file=h)

        # Transmission coefficients at center frequency
        transmission_center = dataframes['transmission'].iloc[center_idx]
        print(f"\nTransmission Coefficients:")
        print(f"  S21: {transmission_center['S21_dB']:.1f} dB", file=h)
        print(f"  S31: {transmission_center['S31_dB']:.1f} dB", file=h)
        print(f"  S32: {transmission_center['S32_dB']:.1f} dB", file=h)

        # Find worst isolation frequency
        isolation_df = dataframes['isolation']
        worst_isolation_idx = isolation_df[['Isolation_Ant1_2_dB', 'Isolation_Ant1_3_dB', 'Isolation_Ant2_3_dB']].min().idxmin()
        worst_value = isolation_df[worst_isolation_idx].min()
        worst_freq = isolation_df.loc[isolation_df[worst_isolation_idx].idxmin(), 'Frequency_GHz']

        print(f"\nWorst Isolation: {worst_value:.1f} dB at {worst_freq:.3f} GHz", file=h)
        print(f"Parameter: {worst_isolation_idx}", file=h)

        plt.show()

if __name__ == "__main__":
    main()

;;;;
%utl_slc_pyendx

/*___                _   _                              _               _
| ___|   _ __  _   _| |_| |__   ___  _ __    ___  _   _| |_ _ __  _   _| |_ ___
|___ \  | `_ \| | | | __| `_ \ / _ \| `_ \  / _ \| | | | __| `_ \| | | | __/ __|
 ___) | | |_) | |_| | |_| | | | (_) | | | || (_) | |_| | |_| |_) | |_| | |_\__ \
|____/  | .__/ \__, |\__|_| |_|\___/|_| |_| \___/ \__,_|\__| .__/ \__,_|\__|___/
        |_|    |___/   _                                   |_|
  __ _ _ __ __ _ _ __ | |__
 / _` | `__/ _` | `_ \| `_ \
| (_| | | | (_| | |_) | | | |
 \__, |_|  \__,_| .__/|_| |_|
 |___/          |_|

d:/lgs/png/cosite_interference_analysis.png
*/

/*                                                                         _
 ___ _   _ _ __ ___  _ __ ___   __ _ _ __ _   _  _ __ ___ _ __   ___  _ __| |_
/ __| | | | `_ ` _ \| `_ ` _ \ / _` | `__| | | || `__/ _ \ `_ \ / _ \| `__| __|
\__ \ |_| | | | | | | | | | | | (_| | |  | |_| || | |  __/ |_) | (_) | |  | |_
|___/\__,_|_| |_| |_|_| |_| |_|\__,_|_|   \__, ||_|  \___| .__/ \___/|_|   \__|
                                          |___/          |_|
d:/lgs/txt/lgs_summary.txt
*/

FREQUENCY DOMAIN DATAFRAMES CREATED:
============================================================

REFLECTION DataFrame:
----------------------------------------
Shape: (501, 4)
Columns: ['Frequency_GHz', 'S11_dB', 'S22_dB', 'S33_dB']

First 5 rows:
   Frequency_GHz     S11_dB     S22_dB     S33_dB
0          2.150 -20.449431 -23.903272 -24.684660
1          2.151 -20.381679 -24.706047 -24.261566
2          2.152 -21.061935 -23.678718 -24.962857
3          2.153 -20.374827 -22.749280 -24.966356
4          2.154 -20.412786 -23.268984 -25.063672

Summary statistics:
       Frequency_GHz      S11_dB      S22_dB      S33_dB
count     501.000000  501.000000  501.000000  501.000000
mean        2.400000  -14.575241  -15.550162  -14.903501
std         0.144771    6.408590    6.509839    6.608689
min         2.150000  -24.015570  -24.706047  -25.063672
25%         2.275000  -19.630593  -21.029778  -19.886112
50%         2.400000  -16.340941  -17.364613  -16.244933
75%         2.525000  -10.494333  -11.592219  -10.556849
max         2.650000    0.905571    0.819249    0.652614

TRANSMISSION DataFrame:
----------------------------------------
Shape: (501, 4)
Columns: ['Frequency_GHz', 'S21_dB', 'S31_dB', 'S32_dB']

First 5 rows:
   Frequency_GHz     S21_dB     S31_dB     S32_dB
0          2.150 -36.610264 -39.080948 -35.049862
1          2.151 -37.112102 -40.013001 -35.801298
2          2.152 -37.207831 -39.865909 -35.101281
3          2.153 -37.258211 -39.984988 -34.607885
4          2.154 -36.721899 -39.376928 -35.032681

Summary statistics:
       Frequency_GHz      S21_dB      S31_dB      S32_dB
count     501.000000  501.000000  501.000000  501.000000
mean        2.400000  -25.815644  -28.304982  -23.877094
std         0.144771    5.252102    5.217447    5.255423
min         2.150000  -37.892321  -40.013001  -36.379721
25%         2.275000  -29.695111  -32.244178  -27.948517
50%         2.400000  -24.325674  -26.932665  -22.414843
75%         2.525000  -21.109677  -23.649432  -19.191179
max         2.650000  -19.191672  -21.476519  -17.174200

ISOLATION DataFrame:
----------------------------------------
Shape: (501, 4)
Columns: ['Frequency_GHz', 'Isolation_Ant1_2_dB', 'Isolation_Ant1_3_dB', 'Isolation_Ant2_3_dB']

First 5 rows:
   Frequency_GHz  Isolation_Ant1_2_dB  Isolation_Ant1_3_dB  Isolation_Ant2_3_dB
0          2.150            36.610264            39.080948            35.049862
1          2.151            37.112102            40.013001            35.801298
2          2.152            37.207831            39.865909            35.101281
3          2.153            37.258211            39.984988            34.607885
4          2.154            36.721899            39.376928            35.032681

Summary statistics:
       Frequency_GHz  ...  Isolation_Ant2_3_dB
count     501.000000  ...           501.000000
mean        2.400000  ...            23.877094
std         0.144771  ...             5.255423
min         2.150000  ...            17.174200
25%         2.275000  ...            19.191179
50%         2.400000  ...            22.414843
75%         2.525000  ...            27.948517
max         2.650000  ...            36.379721

[8 rows x 4 columns]

TOTAL_COUPLING DataFrame:
----------------------------------------
Shape: (501, 4)
Columns: ['Frequency_GHz', 'Total_Coupling_Ant1_dB', 'Total_Coupling_Ant2_dB', 'Total_Coupling_Ant3_dB']

First 5 rows:
   Frequency_GHz  ...  Total_Coupling_Ant3_dB
0          2.150  ...              -34.094120
1          2.151  ...              -34.646212
2          2.152  ...              -34.102019
3          2.153  ...              -33.726905
4          2.154  ...              -33.647274

[5 rows x 4 columns]

Summary statistics:
       Frequency_GHz  ...  Total_Coupling_Ant3_dB
count     501.000000  ...              501.000000
mean        2.400000  ...              -22.516308
std         0.144771  ...                5.227511
min         2.150000  ...              -34.646212
25%         2.275000  ...              -26.550142
50%         2.400000  ...              -20.961267
75%         2.525000  ...              -17.928168
max         2.650000  ...              -15.906657

[8 rows x 4 columns]

VSWR_SUMMARY DataFrame:
----------------------------------------
Shape: (3, 4)
Columns: ['Antenna', 'VSWR_Min', 'VSWR_Max', 'VSWR_Center']
     Antenna    VSWR_Min    VSWR_Max  VSWR_Center
0  Antenna 1 -105.928704  402.257540     2.218064
1  Antenna 2 -464.252931  336.495521    27.753743
2  Antenna 3 -209.435792  106.653729     1.923473

Summary statistics:

ISOLATION_SUMMARY DataFrame:
----------------------------------------
Shape: (3, 5)
Columns: ['Antenna_Pair', 'S_Parameter', 'Min_Isolation_dB', 'Avg_Isolation_dB', 'Worst_Freq_GHz']
  Antenna_Pair S_Parameter  Min_Isolation_dB  Avg_Isolation_dB  Worst_Freq_GHz
0          1-2         S21         19.191672         23.714089           2.399
1          1-3         S31         21.476519         26.202415           2.391
2          2-3         S32         17.174200         21.778036           2.406

Summary statistics:

S_MATRIX DataFrame:
----------------------------------------
Shape: (9, 5)
Columns: ['Input_Port', 'Output_Port', 'S_Parameter', 'Magnitude_dB', 'Frequency_GHz']

First 5 rows:
  Input_Port Output_Port S_Parameter  Magnitude_dB  Frequency_GHz
0     Port 1      Port 1         S11     -8.438491            2.4
1     Port 2      Port 1         S12    -19.244423            2.4
2     Port 3      Port 1         S13    -23.224363            2.4
3     Port 1      Port 2         S21    -20.006044            2.4
4     Port 2      Port 2         S22     -0.626197            2.4

Summary statistics:
       Magnitude_dB  Frequency_GHz
count      9.000000            9.0
mean     -15.528501            2.4
std        7.512871            0.0
min      -23.224363            2.4
25%      -20.006044            2.4
50%      -18.272181            2.4
75%      -10.009497            2.4
max       -0.626197            2.4

Analysis complete! Results saved to 'd:/lgs/png/cosite_interference_analysis.png'

Additional Metrics:
----------------------------------------
Minimum S21 (Ant1?Ant2): -37.9 dB
Minimum S31 (Ant1?Ant3): -40.0 dB
Minimum S32 (Ant2?Ant3): -36.4 dB

Average Isolation in Operating Band:
S21 average: -20.2 dB
S31 average: -22.8 dB
S32 average: -18.3 dB

============================================================
EXAMPLE DATA ACCESS FROM DATAFRAMES:
============================================================

Data at center frequency (2.40 GHz):
----------------------------------------
Reflection Coefficients:
  S11: -8.4 dB
  S22: -0.6 dB
  S33: -10.0 dB
  S21: -20.0 dB
  S31: -22.1 dB
  S32: -18.3 dB

Worst Isolation: 17.2 dB at 2.406 GHz
Parameter: Isolation_Ant2_3_dB

/*      _                                        _      __ _ _
 _ __  | |_ _ __ __ _ _ __  ___ _ __   ___  _ __| |_   / _(_) | ___  ___
| `__| | __| `__/ _` | `_ \/ __| `_ \ / _ \| `__| __| | |_| | |/ _ \/ __|
| |    | |_| | | (_| | | | \__ \ |_) | (_) | |  | |_  |  _| | |  __/\__ \
|_|     \__|_|  \__,_|_| |_|___/ .__/ \___/|_|   \__| |_| |_|_|\___||___/
                               |_|

d:/lgs/rds/isolation_data.rds
d:/lgs/rds/reflection_data.rds
d:/lgs/rds/total_coupling_data.rds
d:/lgs/rds/transmission_data.rds

*/

/*__
 / /_    _ __   _ __  _ __ ___   ___ ___  ___ ___
| `_ \  | `__| | `_ \| `__/ _ \ / __/ _ \/ __/ __|
| (_) | | |    | |_) | | | (_) | (_|  __/\__ \__ \
 \___/  |_|    | .__/|_|  \___/ \___\___||___/___/
               |_|
*/

libname lgs sas7bdat "d:/lgs";
options set=RHOME 'D:\r414';
proc r;
submit;
isolation     <-readRDS('d:/lgs/rds/isolation_data.rds')
reflection    <-readRDS('d:/lgs/rds/reflection_data.rds')
total_coupling<-readRDS('d:/lgs/rds/total_coupling_data.rds')
transmission  <-readRDS('d:/lgs/rds/transmission_data.rds')
endsubmit;
import data=lgs.isolation       r=isolation     ;
import data=lgs.reflection      r=reflection    ;
import data=lgs.total_coupling  r=total_coupling;
import data=lgs.transmission    r=transmission  ;
;quit;run;

/*____                      _               _
|___  |  _ __    ___  _   _| |_ _ __  _   _| |_
   / /  | `__|  / _ \| | | | __| `_ \| | | | __|
  / /   | |    | (_) | |_| | |_| |_) | |_| | |_
 /_/    |_|     \___/ \__,_|\__| .__/ \__,_|\__|
     _        _        _     _ |_|
 ___| | ___  | |_ __ _| |__ | | ___  ___
/ __| |/ __| | __/ _` | `_ \| |/ _ \/ __|
\__ \ | (__  | || (_| | |_) | |  __/\__ \
|___/_|\___|  \__\__,_|_.__/|_|\___||___/

*/

libname lgs sas7bdat  "d:/lgs";

%macro fiveobs(dsn);
  proc print data=&dsn(obs=5);
   TITLE "*--- FIRST 5 OBS FROM &DSN ---*";
  run;quit;
%mend fiveobs;

%fiveobs(lgs.isolation     );
%fiveobs(lgs.reflection    );
%fiveobs(lgs.total_coupling);
%fiveobs(lgs.transmission  );

/*---

*--- FIRST 5 OBS FROM lgs.isolation ---*;

       FREQUENCY_    ISOLATION_    ISOLATION_    ISOLATION_
Obs        GHZ        ANT1_2_DB     ANT1_3_DB     ANT2_3_DB

  1       2.150        36.6103       39.0809       35.0499
  2       2.151        37.1121       40.0130       35.8013
  3       2.152        37.2078       39.8659       35.1013
  4       2.153        37.2582       39.9850       34.6079
  5       2.154        36.7219       39.3769       35.0327


*--- FIRST 5 OBS FROM lgs.reflection ---*;

       FREQUENCY_
Obs        GHZ        S11_DB      S22_DB      S33_DB

  1       2.150      -20.4494    -23.9033    -24.6847
  2       2.151      -20.3817    -24.7060    -24.2616
  3       2.152      -21.0619    -23.6787    -24.9629
  4       2.153      -20.3748    -22.7493    -24.9664
  5       2.154      -20.4128    -23.2690    -25.0637


*--- FIRST 5 OBS FROM lgs.total_coupling ---*;

                       TOTAL_       TOTAL_       TOTAL_
       FREQUENCY_    COUPLING_    COUPLING_    COUPLING_
Obs        GHZ        ANT1_DB      ANT2_DB      ANT3_DB

  1       2.150       -34.6619     -33.0726     -34.0941
  2       2.151       -35.3144     -33.5436     -34.6462
  3       2.152       -35.3263     -33.2344     -34.1020
  4       2.153       -35.4007     -32.7234     -33.7269
  5       2.154       -34.8393     -33.1893     -33.6473


*--- FIRST 5 OBS FROM lgs.transmission ---*;

       FREQUENCY_
Obs        GHZ        S21_DB      S31_DB      S32_DB

  1       2.150      -36.6103    -39.0809    -35.0499
  2       2.151      -37.1121    -40.0130    -35.8013
  3       2.152      -37.2078    -39.8659    -35.1013
  4       2.153      -37.2582    -39.9850    -34.6079
  5       2.154      -36.7219    -39.3769    -35.0327
---*/

/*___                _     _                _
 ( _ )   _ __  _   _| |__ | |_ ___  _ __   | | ___   __ _
 / _ \  | `_ \| | | | `_ \| __/ _ \| `_ \  | |/ _ \ / _` |
| (_) | | |_) | |_| | | | | || (_) | | | | | | (_) | (_| |
 \___/  | .__/ \__, |_| |_|\__\___/|_| |_| |_|\___/ \__, |
        |_|    |___/                                |___/
*/

1                                          Altair SLC     11:35 Thursday, December 18, 2025

NOTE: Copyright 2002-2025 World Programming, an Altair Company
NOTE: Altair SLC 2026 (05.26.01.00.000758)
      Licensed to Roger DeAngelis
NOTE: This session is executing on the X64_WIN11PRO platform and is running in 64 bit mode

NOTE: AUTOEXEC processing beginning; file is C:\wpsoto\autoexec.sas
NOTE: AUTOEXEC source line
1       +  ?ods _all_ close;
           ^
ERROR: Expected a statement keyword : found "?"
NOTE: Library workx assigned as follows:
      Engine:        SAS7BDAT
      Physical Name: d:\wpswrkx


NOTE: 1 record was written to file PRINT

NOTE: The data step took :
      real time : 0.031
      cpu time  : 0.015


NOTE: AUTOEXEC processing completed

1          /*--- Coupling coefficients between antennas (normalized 0-1) ---*/
2         %let coupling_matrix=[[1.0, 0.2, 0.15],[0.2, 1.0, 0.25],[0.15, 0.25, 1.0]];
3
4         /*---
5         coupling_matrix
6         [1.0, 0.2, 0.15],    # S11, S21, S31
7         [0.2, 1.0, 0.25],    # S12, S22, S32
8         [0.15, 0.25, 1.0]    # S13, S23, S33
9         ---*/
10
11        %let center_freq=2.4e9;
12
13        %let bandwidth=500e6;
14        %let Q_factors = [30, 35, 32];
15
16        %let noise_floor = -90;
17
18        /*--- Antenna resonant frequencies (slightly offset for realism) ---*/
19
20        %let  center_freq1 =  0.98;
21        /*--- center_freq2 no offset ---*/
22        %let  center_freq3 =  1.02;
23
24        /*  _                 _   _
25        | || |    _ __  _   _| |_| |__   ___  _ __   _ __  _ __ ___   ___ ___  ___ ___
26        | || |_  | `_ \| | | | __| `_ \ / _ \| `_ \ | `_ \| `__/ _ \ / __/ _ \/ __/ __|
27        |__   _| | |_) | |_| | |_| | | | (_) | | | || |_) | | | (_) | (_|  __/\__ \__ \
28           |_|   | .__/ \__, |\__|_| |_|\___/|_| |_|| .__/|_|  \___/ \___\___||___/___/
29                 |_|    |___/                       |_|
30        */
31
32        libname lgs sas7bdat  "d:/lgs";
NOTE: Library lgs assigned as follows:
      Engine:        SAS7BDAT
      Physical Name: d:\lgs

33
34        /*--- slc output tables                    ---*/
35

2                                                                                                                         Altair SLC

36
37        Proc delete data=
38           lgs.isolation
39           lgs.reflection
40           lgs.total_coupling
41           lgs.transmission
42           ;
43        ;run;quit;
NOTE: LGS.ISOLATION (memtype="DATA") was not found, and has not been deleted
NOTE: LGS.REFLECTION (memtype="DATA") was not found, and has not been deleted
NOTE: LGS.TOTAL_COUPLING (memtype="DATA") was not found, and has not been deleted
NOTE: LGS.TRANSMISSION (memtype="DATA") was not found, and has not been deleted
NOTE: Procedure delete step took :
      real time : 0.000
      cpu time  : 0.000


44
45        /*--- CSVs BEDIND ALL FOUR FREQUENCY PLOTS ---*/
46
47        %utlfkil(d:/lgs/csv/isolation_data.csv);
The file d:/lgs/csv/isolation_data.csv does not exist
48        %utlfkil(d:/lgs/csv/reflection_data.csv);
The file d:/lgs/csv/reflection_data.csv does not exist
49        %utlfkil(d:/lgs/csv/total_coupling_data.csv);
The file d:/lgs/csv/total_coupling_data.csv does not exist
50        %utlfkil(vd:/lgs/csv/transmission_data.csv);
The file vd:/lgs/csv/transmission_data.csv does not exist
51
52        /*--- R TRANSPOT TABLES CREATED BY PYTHON  ---*/
53
54        %utlfkil(d:/lgs/rds/isolation_data.rds);
The file d:/lgs/rds/isolation_data.rds does not exist
55        %utlfkil(d:/lgs/rds/reflection_data.rds);
The file d:/lgs/rds/reflection_data.rds does not exist
56        %utlfkil(d:/lgs/rds/total_coupling_data.rds);
The file d:/lgs/rds/total_coupling_data.rds does not exist
57        %utlfkil(d:/lgs/rds/transmission_data.rds);
The file d:/lgs/rds/transmission_data.rds does not exist
58
59        %utlfkil(d:/lgs/txt/lgs_summary.txt);
The file d:/lgs/txt/lgs_summary.txt does not exist
60
61        %utlfkil(d:/png/cosite_interference_analysis.png);
The file d:/png/cosite_interference_analysis.png does not exist
62
63        %utl_slc_pybeginx(
64           return=NA      /*- NA=disabled / return clipboard as maco var   -*/
65          ,resolve=Y      /*- NA=disabled / resolve macros in python       -*/
66          ,in=NA          /*- NA=disabled / input sas dataset              -*/
67          ,out=NA         /*- NA=disabled / output work.female             -*/
68          ,py2r=NA );     /*- NA=disabled / temp/rds,rds                   -*/
The file c:/temp/py_dataframe.rds does not exist
The file c:/temp/py_procr.sas does not exist
69        cards4;

NOTE: The file 'c:\temp\py_pgm.py' is:
      Filename='c:\temp\py_pgm.py',
      Owner Name=T7610\Roger,
      File size (bytes)=0,
      Create Time=10:08:39 Dec 18 2025,
      Last Accessed=11:35:24 Dec 18 2025,
      Last Modified=11:35:24 Dec 18 2025,

3                                                                                                                         Altair SLC

      Lrecl=32767, Recfm=V

NOTE: The file 'c:\temp\py_mac.sas' is:
      Filename='c:\temp\py_mac.sas',
      Owner Name=T7610\Roger,
      File size (bytes)=0,
      Create Time=10:08:39 Dec 18 2025,
      Last Accessed=11:35:24 Dec 18 2025,
      Last Modified=11:35:24 Dec 18 2025,
      Lrecl=32767, Recfm=V

NOTE: 578 records were written to file 'c:\temp\py_pgm.py'
      The minimum record length was 80
      The maximum record length was 152
NOTE: 1 record was written to file 'c:\temp\py_mac.sas'
      The minimum record length was 15
      The maximum record length was 15
NOTE: The data step took :
      real time : 0.006
      cpu time  : 0.015


70        import numpy as np
71        import matplotlib.pyplot as plt
72        from scipy import signal
73        import pandas as pd
74        import sys
75        from matplotlib.gridspec import GridSpec
76        from datetime import date
77        import pyreadr as pr
78
79        # Set random seed for reproducibility
80        np.random.seed(42)
81
82        class ThreeAntennaSystem:
83            """Simulate a three-antenna system with mutual coupling"""
84
85            def __init__(self, center_freq=&center_freq, bandwidth=&bandwidth):
86                """
87                Initialize the antenna system
88                center_freq: Center frequency in Hz (default: 2.4 GHz)
89                bandwidth: Bandwidth in Hz (default: 500 MHz)
90                """
91                self.center_freq = center_freq
92                self.bandwidth = bandwidth
93
94                # Antenna resonant frequencies (slightly offset for realism)
95                self.f_res = [
96                    center_freq * &center_freq1,  # Antenna 1
97                    center_freq,         # Antenna 2
98                    center_freq * &center_freq3   # Antenna 3
99                ]
100
101               # Quality factors for each antenna
102               self.Q_factors = &Q_factors
103
104               # Coupling coefficients between antennas (normalized 0-1)
105               #self.coupling_matrix = np.array([
106               #    [1.0, 0.2, 0.15],    # S11, S21, S31
107               #    [0.2, 1.0, 0.25],    # S12, S22, S32
108               #    [0.15, 0.25, 1.0]    # S13, S23, S33
109               #])
110

4                                                                                                                         Altair SLC

111               self.coupling_matrix = np.array(&coupling_matrix)
112
113               # Noise floor for realistic simulation
114               self.noise_floor = &noise_floor  # dB
115
116           def generate_s_parameters(self, freq_points=501):
117               """Generate S-parameters across frequency range"""
118               frequencies = np.linspace(
119                   self.center_freq - self.bandwidth/2,
120                   self.center_freq + self.bandwidth/2,
121                   freq_points
122               )
123
124               # Initialize S-parameter matrix: [freq, port_out, port_in]
125               s_params = np.zeros((len(frequencies), 3, 3), dtype=complex)
126
127               # Generate self-impedance (reflection coefficients)
128               for i in range(3):
129                   for j, freq in enumerate(frequencies):
130                       # Resonant response for reflection coefficients
131                       omega = 2 * np.pi * freq
132                       omega_res = 2 * np.pi * self.f_res[i]
133
134                       # Second-order resonant response
135                       s_mag = 1 / np.sqrt(1 + (2 * self.Q_factors[i] *
136                                               (freq/self.f_res[i] - self.f_res[i]/freq))**2)
137
138                       # Add phase shift
139                       phase = -2 * np.arctan(2 * self.Q_factors[i] *
140                                             (freq/self.f_res[i] - 1))
141
142                       s_params[j, i, i] = s_mag * np.exp(1j * phase)
143
144               # Generate mutual coupling (transmission coefficients)
145               for out_port in range(3):
146                   for in_port in range(3):
147                       if out_port != in_port:
148                           coupling = self.coupling_matrix[out_port, in_port]
149
150                           # Frequency-dependent coupling (bandpass characteristic)
151                           for j, freq in enumerate(frequencies):
152                               # Coupling decreases away from center frequency
153                               freq_factor = np.exp(-((freq - self.center_freq)**2) /
154                                                  (2 * (self.bandwidth/4)**2))
155
156                               # Transmission coefficient with realistic phase
157                               s_mag = coupling * freq_factor * 0.5
158                               phase = -2 * np.pi * np.abs(out_port - in_port) * freq / 3e8 * 0.1
159
160                               s_params[j, out_port, in_port] = s_mag * np.exp(1j * phase)
161
162               # Convert to dB and add realistic noise
163               s_params_db = 20 * np.log10(np.abs(s_params) + 1e-10)
164               noise = np.random.normal(0, 0.5, s_params_db.shape)  # 0.5 dB noise
165               s_params_db += noise
166               s_params_db = np.maximum(s_params_db, self.noise_floor)
167
168               return frequencies, s_params_db
169
170           def create_frequency_dataframes(self, frequencies, s_params_db, results):
171               """Create pandas DataFrames for frequency domain data"""
172
173               freq_ghz = frequencies / 1e9

5                                                                                                                         Altair SLC

174
175               # DataFrame 1: Reflection Coefficients
176               reflection_df = pd.DataFrame({
177                   'Frequency_GHz': freq_ghz,
178                   'S11_dB': s_params_db[:, 0, 0],
179                   'S22_dB': s_params_db[:, 1, 1],
180                   'S33_dB': s_params_db[:, 2, 2]
181               })
182
183               # DataFrame 2: Transmission Coefficients (Coupling)
184               transmission_df = pd.DataFrame({
185                   'Frequency_GHz': freq_ghz,
186                   'S21_dB': s_params_db[:, 1, 0],  # Port 1 to Port 2
187                   'S31_dB': s_params_db[:, 2, 0],  # Port 1 to Port 3
188                   'S32_dB': s_params_db[:, 2, 1]   # Port 2 to Port 3
189               })
190
191               # DataFrame 3: Isolation (negative of coupling)
192               isolation_df = pd.DataFrame({
193                   'Frequency_GHz': freq_ghz,
194                   'Isolation_Ant1_2_dB': -s_params_db[:, 1, 0],  # Negative of S21
195                   'Isolation_Ant1_3_dB': -s_params_db[:, 2, 0],  # Negative of S31
196                   'Isolation_Ant2_3_dB': -s_params_db[:, 2, 1]   # Negative of S32
197               })
198
199               # DataFrame 4: Total Coupling to each antenna
200               total_coupling_df = pd.DataFrame({
201                   'Frequency_GHz': freq_ghz,
202                   'Total_Coupling_Ant1_dB': results['total_coupling']['Antenna 1'],
203                   'Total_Coupling_Ant2_dB': results['total_coupling']['Antenna 2'],
204                   'Total_Coupling_Ant3_dB': results['total_coupling']['Antenna 3']
205               })
206
207               # DataFrame 5: VSWR Summary
208               vswr_data = []
209               for i in range(3):
210                   vswr_data.append({
211                       'Antenna': f'Antenna {i+1}',
212                       'VSWR_Min': results['vswr'][f'Antenna {i+1}']['min'],
213                       'VSWR_Max': results['vswr'][f'Antenna {i+1}']['max'],
214                       'VSWR_Center': results['vswr'][f'Antenna {i+1}']['center']
215                   })
216               vswr_summary_df = pd.DataFrame(vswr_data)
217
218               # DataFrame 6: Isolation Summary
219               isolation_summary_data = []
220               for iso_data in results['isolation']:
221                   isolation_summary_data.append({
222                       'Antenna_Pair': iso_data['pair'],
223                       'S_Parameter': iso_data['label'],
224                       'Min_Isolation_dB': iso_data['min_isolation'],
225                       'Avg_Isolation_dB': iso_data['avg_isolation'],
226                       'Worst_Freq_GHz': iso_data['worst_freq']
227                   })
228               isolation_summary_df = pd.DataFrame(isolation_summary_data)
229
230               # DataFrame 7: S-Parameter Matrix at Center Frequency
231               center_idx = len(frequencies) // 2
232               s_matrix_data = []
233               for i in range(3):
234                   for j in range(3):
235                       s_matrix_data.append({
236                           'Input_Port': f'Port {j+1}',

6                                                                                                                         Altair SLC

237                           'Output_Port': f'Port {i+1}',
238                           'S_Parameter': f'S{i+1}{j+1}',
239                           'Magnitude_dB': s_params_db[center_idx, i, j],
240                           'Frequency_GHz': freq_ghz[center_idx]
241                       })
242               s_matrix_df = pd.DataFrame(s_matrix_data)
243
244               return {
245                   'reflection': reflection_df,
246                   'transmission': transmission_df,
247                   'isolation': isolation_df,
248                   'total_coupling': total_coupling_df,
249                   'vswr_summary': vswr_summary_df,
250                   'isolation_summary': isolation_summary_df,
251                   's_matrix': s_matrix_df
252               }
253
254           def calculate_isolation(self, s_params_db):
255               """Calculate isolation metrics from S-parameters"""
256               # Minimum isolation (worst-case coupling)
257               s21_min = np.min(s_params_db[:, 1, 0])
258               s31_min = np.min(s_params_db[:, 2, 0])
259               s32_min = np.min(s_params_db[:, 2, 1])
260
261               # Average isolation in operating band
262               center_idx = len(s_params_db) // 2
263               band_idx = slice(center_idx - 50, center_idx + 50)
264
265               avg_isolation = {
266                   'S21_avg': np.mean(s_params_db[band_idx, 1, 0]),
267                   'S31_avg': np.mean(s_params_db[band_idx, 2, 0]),
268                   'S32_avg': np.mean(s_params_db[band_idx, 2, 1])
269               }
270
271               return s21_min, s31_min, s32_min, avg_isolation
272
273           def analyze_cosite_interference(self, frequencies, s_params_db):
274               """Perform co-site interference analysis"""
275               results = {}
276
277               # 1. Reflection coefficient analysis (match quality)
278               results['vswr'] = {}
279               for i in range(3):
280                   s11 = s_params_db[:, i, i]
281                   # Calculate VSWR from S11
282                   gamma = 10**(s11/20)
283                   vswr = (1 + gamma) / (1 - gamma)
284                   results['vswr'][f'Antenna {i+1}'] = {
285                       'min': np.min(vswr),
286                       'max': np.max(vswr),
287                       'center': vswr[len(vswr)//2]
288                   }
289
290               # 2. Isolation analysis
291               isolation_data = []
292               pairs = [(0, 1, 'S21'), (0, 2, 'S31'), (1, 2, 'S32')]
293
294               for i, j, label in pairs:
295                   coupling = s_params_db[:, j, i]
296                   isolation = -coupling  # Isolation is negative of coupling in dB
297                   isolation_data.append({
298                       'pair': f'{i+1}-{j+1}',
299                       'label': label,

7                                                                                                                         Altair SLC

300                       'min_isolation': np.min(isolation),
301                       'avg_isolation': np.mean(isolation[50:-50]),
302                       'worst_freq': frequencies[np.argmin(isolation)] / 1e9
303                   })
304
305               results['isolation'] = isolation_data
306
307               # 3. Calculate total coupling to each antenna
308               results['total_coupling'] = {}
309               for i in range(3):
310                   total_power = 0
311                   for j in range(3):
312                       if i != j:
313                           # Convert from dB to linear, sum power, convert back to dB
314                           power_linear = 10**(s_params_db[:, j, i]/10)
315                           total_power += power_linear
316                   results['total_coupling'][f'Antenna {i+1}'] = 10 * np.log10(total_power + 1e-10)
317
318               return results
319
320       def plot_s_parameters(frequencies, s_params_db, results, dataframes=None):
321           """Create comprehensive visualization of S-parameters"""
322           fig = plt.figure(figsize=(16, 12))
323           gs = GridSpec(3, 3, figure=fig)
324
325           # Convert frequencies to GHz for plotting
326           freq_ghz = frequencies / 1e9
327
328           # Plot 1: Reflection coefficients (S11, S22, S33)
329           ax1 = fig.add_subplot(gs[0, 0])
330           colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
331           for i in range(3):
332               ax1.plot(freq_ghz, s_params_db[:, i, i],
333                       label=f'S{i+1}{i+1}', color=colors[i], linewidth=2)
334
335           ax1.set_xlabel('Frequency (GHz)')
336           ax1.set_ylabel('Magnitude (dB)')
337           ax1.set_title('Reflection Coefficients')
338           ax1.grid(True, alpha=0.3)
339           ax1.legend()
340           ax1.set_ylim([-30, 0])
341
342           # Plot 2: Transmission coefficients (coupling)
343           ax2 = fig.add_subplot(gs[0, 1])
344           coupling_pairs = [(0, 1, 'S21'), (0, 2, 'S31'), (1, 2, 'S32')]
345           for i, j, label in coupling_pairs:
346               ax2.plot(freq_ghz, s_params_db[:, j, i],
347                       label=label, linewidth=2)
348
349           ax2.set_xlabel('Frequency (GHz)')
350           ax2.set_ylabel('Magnitude (dB)')
351           ax2.set_title('Transmission Coefficients (Coupling)')
352           ax2.grid(True, alpha=0.3)
353           ax2.legend()
354           ax2.set_ylim([-50, -10])
355
356           # Plot 3: Isolation between antenna pairs
357           ax3 = fig.add_subplot(gs[0, 2])
358           for i, j, label in coupling_pairs:
359               isolation = -s_params_db[:, j, i]  # Isolation is negative of coupling
360               ax3.plot(freq_ghz, isolation,
361                       label=f'Ant {i+1}-{j+1}', linewidth=2)
362

8                                                                                                                         Altair SLC

363           ax3.set_xlabel('Frequency (GHz)')
364           ax3.set_ylabel('Isolation (dB)')
365           ax3.set_title('Antenna-to-Antenna Isolation')
366           ax3.grid(True, alpha=0.3)
367           ax3.legend()
368           ax3.set_ylim([10, 50])
369
370           # Plot 4: 3D visualization of coupling matrix at center frequency
371           ax4 = fig.add_subplot(gs[1, :])
372           center_idx = len(frequencies) // 2
373
374           # Create coupling matrix heatmap
375           coupling_matrix = np.zeros((3, 3))
376           for i in range(3):
377               for j in range(3):
378                   coupling_matrix[i, j] = s_params_db[center_idx, j, i]
379
380           im = ax4.imshow(coupling_matrix, cmap='viridis', aspect='auto')
381           ax4.set_title(f'S-parameter Matrix at {freq_ghz[center_idx]:.2f} GHz')
382           ax4.set_xticks([0, 1, 2])
383           ax4.set_yticks([0, 1, 2])
384           ax4.set_xticklabels(['Port 1', 'Port 2', 'Port 3'])
385           ax4.set_yticklabels(['Port 1', 'Port 2', 'Port 3'])
386
387           # Add text annotations
388           for i in range(3):
389               for j in range(3):
390                   text = ax4.text(j, i, f'{coupling_matrix[i, j]:.1f} dB',
391                                 ha="center", va="center",
392                                 color="white" if coupling_matrix[i, j] < -25 else "black")
393
394           plt.colorbar(im, ax=ax4, label='Magnitude (dB)')
395
396           # Plot 5: Total coupling to each antenna
397           ax5 = fig.add_subplot(gs[2, 0])
398           for i in range(3):
399               total_coupling = results['total_coupling'][f'Antenna {i+1}']
400               ax5.plot(freq_ghz, total_coupling,
401                       label=f'Antenna {i+1}', linewidth=2)
402
403           ax5.set_xlabel('Frequency (GHz)')
404           ax5.set_ylabel('Total Coupling (dB)')
405           ax5.set_title('Total Interference Coupling per Antenna')
406           ax5.grid(True, alpha=0.3)
407           ax5.legend()
408           ax5.set_ylim([-40, -10])
409
410           # Plot 6: VSWR analysis
411           ax6 = fig.add_subplot(gs[2, 1])
412           antenna_labels = []
413           vswr_center = []
414
415           for i in range(3):
416               antenna_labels.append(f'Ant {i+1}')
417               vswr_center.append(results['vswr'][f'Antenna {i+1}']['center'])
418
419           bars = ax6.bar(antenna_labels, vswr_center, color=colors)
420           ax6.set_ylabel('VSWR at Center Frequency')
421           ax6.set_title('Voltage Standing Wave Ratio')
422           ax6.grid(True, alpha=0.3, axis='y')
423
424           # Add VSWR threshold lines
425           ax6.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='VSWR=2.0')

9                                                                                                                         Altair SLC

426           ax6.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='VSWR=1.5')
427           ax6.legend()
428
429           # Plot 7: Isolation summary
430           ax7 = fig.add_subplot(gs[2, 2])
431           isolation_pairs = []
432           min_isolation_values = []
433
434           for iso_data in results['isolation']:
435               isolation_pairs.append(iso_data['pair'])
436               min_isolation_values.append(iso_data['min_isolation'])
437
438           bars = ax7.bar(isolation_pairs, min_isolation_values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
439           ax7.set_ylabel('Minimum Isolation (dB)')
440           ax7.set_title('Worst-case Antenna Isolation')
441           ax7.grid(True, alpha=0.3, axis='y')
442
443           # Add isolation requirement line
444           ax7.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20 dB Requirement')
445           ax7.legend()
446
447           # Add value labels on bars
448           for bar, val in zip(bars, min_isolation_values):
449               height = bar.get_height()
450               ax7.text(bar.get_x() + bar.get_width()/2., height,
451                       f'{val:.1f} dB', ha='center', va='bottom')
452
453           plt.suptitle('Three-Antenna Co-site Interference Analysis', fontsize=16, fontweight='bold')
454           plt.tight_layout()
455
456           return fig
457
458       def generate_report(results, frequencies, s_params_db, dataframes):
459           """Generate a text report of the analysis"""
460
461           with open("d:/lgs/txt/lgs_summary.txt", 'a') as g:
462               print("Starting Co-site Interference Analysis for Three-Antenna System...", file=g)
463               print("="*70, file=g)
464               print("CO-SITE INTERFERENCE ANALYSIS REPORT", file=g)
465               print("="*70, file=g)
466
467               print("\n1. REFLECTION COEFFICIENT SUMMARY (S11, S22, S33):", file=g)
468               print("-"*50, file=g)
469               for i in range(3):
470                   vswr_data = results['vswr'][f'Antenna {i+1}']
471                   print(f"Antenna {i+1} (S{i+1}{i+1}):", file=g)
472                   print(f"  - VSWR range: {vswr_data['min']:.2f} - {vswr_data['max']:.2f}", file=g)
473                   print(f"  - VSWR at center: {vswr_data['center']:.2f}", file=g)
474                   print(f"  - Match quality: {'Good' if vswr_data['center'] < 2.0 else 'Marginal' if vswr_data['center'] < 3.0 else 'Poor'}", file=g)
475
476               print("\n2. ISOLATION ANALYSIS:", file=g)
477               print("-"*50, file=g)
478               for iso_data in results['isolation']:
479                   print(f"{iso_data['label']} (Antenna {iso_data['pair'][0]} to {iso_data['pair'][-1]}):", file=g)
480                   print(f"  - Minimum isolation: {iso_data['min_isolation']:.1f} dB", file=g)
481                   print(f"  - Average isolation: {iso_data['avg_isolation']:.1f} dB", file=g)
482                   print(f"  - Worst frequency: {iso_data['worst_freq']:.3f} GHz", file=g)
483                   print(f"  - Status: {'? PASS' if iso_data['min_isolation'] > 20 else '? MARGINAL' if iso_data['min_isolation'] > 15 else '? FAIL'}", file=g)
484
485               print("\n3. DATAFRAME SUMMARIES:", file=g)
486               print("-"*50, file=g)
487               print("DataFrames created for frequency domain analysis:", file=g)
488               for df_name, df in dataframes.items():

10                                                                                                                        Altair SLC

489                   print(f"  - {df_name}: {df.shape[0]} rows × {df.shape[1]} columns", file=g)
490
491               print("\n4. INTERFERENCE ASSESSMENT:", file=g)
492               print("-"*50, file=g)
493
494               # Calculate worst-case scenarios
495               center_idx = len(frequencies) // 2
496               worst_coupling = np.min(s_params_db[:, :, :])
497               worst_pair = np.unravel_index(np.argmin(s_params_db[:, :, :]), s_params_db.shape)
498
499               print(f"Worst-case coupling: {worst_coupling:.1f} dB", file=g)
500               print(f"Occurs at: {frequencies[worst_pair[0]]/1e9:.3f} GHz", file=g)
501               print(f"Between: Port {worst_pair[2]+1} ? Port {worst_pair[1]+1}", file=g)
502
503               print("\n5. RECOMMENDATIONS:", file=g)
504               print("-"*50, file=g)
505
506               # Generate recommendations based on analysis
507               recommendations = []
508
509               # Check isolation requirements
510               isolation_status = all(iso['min_isolation'] > 20 for iso in results['isolation'])
511               if not isolation_status:
512                   recommendations.append("- Consider increasing antenna spacing or adding shielding")
513                   recommendations.append("- Implement frequency separation or filtering")
514
515               # Check VSWR
516               vswr_status = all(v['center'] < 2.0 for v in results['vswr'].values())
517               if not vswr_status:
518                   recommendations.append("- Optimize antenna matching networks")
519
520               if not recommendations:
521                   recommendations.append("- System meets basic co-site requirements")
522                   recommendations.append("- Monitor performance with actual hardware")
523
524               for rec in recommendations:
525                   print(rec, file=g)
526
527               print("\n" + "="*70, file=g)
528
529               # Save DataFrames to CSV files
530               print("\nSaving DataFrames to CSV files...", file=g)
531               for df_name, df in dataframes.items():
532                   filename = f"d:/lgs/csv/{df_name}_data.csv"
533                   df.to_csv(filename, index=False)
534                   print(f"  - Saved {df_name} to {filename}", file=g)
535
536               print("\nSaving DataFrames to R rds files...", file=g)
537               for df_name, df in dataframes.items():
538                   filename = f"d:/lgs/rds/{df_name}_data.rds"
539                   # Correct: Pass the DataFrame 'df', not the name '{df_name}'
540                   pr.write_rds(filename,df)  # Note: parameters swapped - DataFrame first, then filename
541                   print(f"  - Saved {df_name} to {filename}", file=g)
542
543       def main():
544           """Main function to run the analysis"""
545
546           with open("d:/lgs/txt/lgs_summary.txt", 'w') as h:
547
548               # Create antenna system
549               antenna_system = ThreeAntennaSystem(
550                   center_freq=2.4e9,  # 2.4 GHz center frequency
551                   bandwidth=500e6     # 500 MHz bandwidth

11                                                                                                                        Altair SLC

552               )
553
554               # Generate S-parameters
555               frequencies, s_params_db = antenna_system.generate_s_parameters()
556
557               # Perform analysis
558               results = antenna_system.analyze_cosite_interference(frequencies, s_params_db)
559
560               # Create pandas DataFrames for frequency domain data
561               dataframes = antenna_system.create_frequency_dataframes(frequencies, s_params_db, results)
562
563               # Display DataFrame information
564               print("="*60, file=h)
565               print("FREQUENCY DOMAIN DATAFRAMES CREATED:", file=h)
566               print("="*60, file=h)
567
568               for name, df in dataframes.items():
569                   print(f"\n{name.upper()} DataFrame:", file=h)
570                   print("-"*40, file=h)
571                   print(f"Shape: {df.shape}", file=h)
572                   print(f"Columns: {list(df.columns)}", file=h)
573                   if df.shape[0] > 5:
574                       print("\nFirst 5 rows:", file=h)
575                       print(df.head(), file=h)
576                   else:
577                       print(df, file=h)
578                   print("\nSummary statistics:", file=h)
579                   if 'Frequency_GHz' in df.columns and df.shape[0] > 3:
580                       print(df.describe(), file=h)
581
582               # Generate report with DataFrames
583               generate_report(results, frequencies, s_params_db, dataframes)
584
585               # Create visualizations
586               fig = plot_s_parameters(frequencies, s_params_db, results, dataframes)
587
588               # Save results
589               plt.savefig('d:/lgs/png/cosite_interference_analysis.png', dpi=150, bbox_inches='tight')
590               print("\nAnalysis complete! Results saved to 'd:/lgs/png/cosite_interference_analysis.png'", file=h)
591
592               # Display additional metrics
593               print("\nAdditional Metrics:", file=h)
594               print("-"*40, file=h)
595
596               # Calculate and display isolation metrics
597               s21_min, s31_min, s32_min, avg_isolation = antenna_system.calculate_isolation(s_params_db)
598
599               print(f"Minimum S21 (Ant1?Ant2): {s21_min:.1f} dB", file=h)
600               print(f"Minimum S31 (Ant1?Ant3): {s31_min:.1f} dB", file=h)
601               print(f"Minimum S32 (Ant2?Ant3): {s32_min:.1f} dB", file=h)
602
603               print(f"\nAverage Isolation in Operating Band:", file=h)
604               print(f"S21 average: {avg_isolation['S21_avg']:.1f} dB", file=h)
605               print(f"S31 average: {avg_isolation['S31_avg']:.1f} dB", file=h)
606               print(f"S32 average: {avg_isolation['S32_avg']:.1f} dB", file=h)
607
608               # Example: Access specific data from DataFrames
609               print("\n" + "="*60, file=h)
610               print("EXAMPLE DATA ACCESS FROM DATAFRAMES:", file=h)
611               print("="*60, file=h)
612
613               # Get center frequency data
614               center_idx = len(frequencies) // 2

12                                                                                                                        Altair SLC

615               center_freq_ghz = frequencies[center_idx] / 1e9
616
617               print(f"\nData at center frequency ({center_freq_ghz:.2f} GHz):", file=h)
618               print("-"*40, file=h)
619
620               # Reflection coefficients at center frequency
621               reflection_center = dataframes['reflection'].iloc[center_idx]
622               print(f"Reflection Coefficients:", file=h)
623               print(f"  S11: {reflection_center['S11_dB']:.1f} dB", file=h)
624               print(f"  S22: {reflection_center['S22_dB']:.1f} dB", file=h)
625               print(f"  S33: {reflection_center['S33_dB']:.1f} dB", file=h)
626
627               # Transmission coefficients at center frequency
628               transmission_center = dataframes['transmission'].iloc[center_idx]
629               print(f"\nTransmission Coefficients:")
630               print(f"  S21: {transmission_center['S21_dB']:.1f} dB", file=h)
631               print(f"  S31: {transmission_center['S31_dB']:.1f} dB", file=h)
632               print(f"  S32: {transmission_center['S32_dB']:.1f} dB", file=h)
633
634               # Find worst isolation frequency
635               isolation_df = dataframes['isolation']
636               worst_isolation_idx = isolation_df[['Isolation_Ant1_2_dB', 'Isolation_Ant1_3_dB', 'Isolation_Ant2_3_dB']].min().idxmin()
637               worst_value = isolation_df[worst_isolation_idx].min()
638               worst_freq = isolation_df.loc[isolation_df[worst_isolation_idx].idxmin(), 'Frequency_GHz']
639
640               print(f"\nWorst Isolation: {worst_value:.1f} dB at {worst_freq:.3f} GHz", file=h)
641               print(f"Parameter: {worst_isolation_idx}", file=h)
642
643               plt.show()
644
645       if __name__ == "__main__":
646           main()
647
648       ;;;;
649       %utl_slc_pyendx

NOTE: The infile 'c:\temp\py_mac.sas' is:
      Filename='c:\temp\py_mac.sas',
      Owner Name=T7610\Roger,
      File size (bytes)=17,
      Create Time=10:08:39 Dec 18 2025,
      Last Accessed=11:35:24 Dec 18 2025,
      Last Modified=11:35:24 Dec 18 2025,
      Lrecl=32767, Recfm=V

NOTE: 1 record was read from file 'c:\temp\py_mac.sas'
      The minimum record length was 15
      The maximum record length was 15
NOTE: The data step took :
      real time : 0.000
      cpu time  : 0.000


NOTE: WORK.NA (memtype="DATA") was not found, and has not been deleted
NOTE: Procedure datasets step took :
      real time : 0.000
      cpu time  : 0.000



NOTE: The infile 'c:\temp\py_pgm.py' is:
      Filename='c:\temp\py_pgm.py',
      Owner Name=T7610\Roger,

13                                                                                                                        Altair SLC

      File size (bytes)=47904,
      Create Time=10:08:39 Dec 18 2025,
      Last Accessed=11:35:24 Dec 18 2025,
      Last Modified=11:35:24 Dec 18 2025,
      Lrecl=32767, Recfm=V

NOTE: The file 'c:\temp\py_pgm.pyx' is:
      Filename='c:\temp\py_pgm.pyx',
      Owner Name=T7610\Roger,
      File size (bytes)=0,
      Create Time=10:08:39 Dec 18 2025,
      Last Accessed=11:35:24 Dec 18 2025,
      Last Modified=11:35:24 Dec 18 2025,
      Lrecl=32767, Recfm=V

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd
import sys
from matplotlib.gridspec import GridSpec
from datetime import date
import pyreadr as pr

# Set random seed for reproducibility
np.random.seed(42)

class ThreeAntennaSystem:
    """Simulate a three-antenna system with mutual coupling"""

    def __init__(self, center_freq=2.4e9, bandwidth=500e6):
WARNING: End-of-file encountered in string
        """
        Initialize the antenna system
        center_freq: Center frequency in Hz (default: 2.4 GHz)
        bandwidth: Bandwidth in Hz (default: 500 MHz)
WARNING: End-of-file encountered in string
        """
        self.center_freq = center_freq
        self.bandwidth = bandwidth

        # Antenna resonant frequencies (slightly offset for realism)
        self.f_res = [
            center_freq * 0.98,  # Antenna 1
            center_freq,         # Antenna 2
            center_freq * 1.02   # Antenna 3
        ]

        # Quality factors for each antenna
        self.Q_factors = [30, 35, 32]

        # Coupling coefficients between antennas (normalized 0-1)
        #self.coupling_matrix = np.array([
        #    [1.0, 0.2, 0.15],    # S11, S21, S31
        #    [0.2, 1.0, 0.25],    # S12, S22, S32
        #    [0.15, 0.25, 1.0]    # S13, S23, S33
        #])

        self.coupling_matrix = np.array([[1.0, 0.2, 0.15],[0.2, 1.0, 0.25],[0.15, 0.25, 1.0]])

        # Noise floor for realistic simulation
        self.noise_floor = -90  # dB


14                                                                                                                        Altair SLC

    def generate_s_parameters(self, freq_points=501):
        """Generate S-parameters across frequency range"""
        frequencies = np.linspace(
            self.center_freq - self.bandwidth/2,
            self.center_freq + self.bandwidth/2,
            freq_points
        )

        # Initialize S-parameter matrix: [freq, port_out, port_in]
        s_params = np.zeros((len(frequencies), 3, 3), dtype=complex)

        # Generate self-impedance (reflection coefficients)
        for i in range(3):
            for j, freq in enumerate(frequencies):
                # Resonant response for reflection coefficients
                omega = 2 * np.pi * freq
                omega_res = 2 * np.pi * self.f_res[i]

                # Second-order resonant response
                s_mag = 1 / np.sqrt(1 + (2 * self.Q_factors[i] *
                                        (freq/self.f_res[i] - self.f_res[i]/freq))**2)

                # Add phase shift
                phase = -2 * np.arctan(2 * self.Q_factors[i] *
                                      (freq/self.f_res[i] - 1))

                s_params[j, i, i] = s_mag * np.exp(1j * phase)

        # Generate mutual coupling (transmission coefficients)
        for out_port in range(3):
            for in_port in range(3):
                if out_port != in_port:
                    coupling = self.coupling_matrix[out_port, in_port]

                    # Frequency-dependent coupling (bandpass characteristic)
                    for j, freq in enumerate(frequencies):
                        # Coupling decreases away from center frequency
                        freq_factor = np.exp(-((freq - self.center_freq)**2) /
                                           (2 * (self.bandwidth/4)**2))

                        # Transmission coefficient with realistic phase
                        s_mag = coupling * freq_factor * 0.5
                        phase = -2 * np.pi * np.abs(out_port - in_port) * freq / 3e8 * 0.1

                        s_params[j, out_port, in_port] = s_mag * np.exp(1j * phase)

        # Convert to dB and add realistic noise
        s_params_db = 20 * np.log10(np.abs(s_params) + 1e-10)
        noise = np.random.normal(0, 0.5, s_params_db.shape)  # 0.5 dB noise
        s_params_db += noise
        s_params_db = np.maximum(s_params_db, self.noise_floor)

        return frequencies, s_params_db

    def create_frequency_dataframes(self, frequencies, s_params_db, results):
        """Create pandas DataFrames for frequency domain data"""

        freq_ghz = frequencies / 1e9

        # DataFrame 1: Reflection Coefficients
        reflection_df = pd.DataFrame({
            'Frequency_GHz': freq_ghz,
            'S11_dB': s_params_db[:, 0, 0],

15                                                                                                                        Altair SLC

            'S22_dB': s_params_db[:, 1, 1],
            'S33_dB': s_params_db[:, 2, 2]
        })

        # DataFrame 2: Transmission Coefficients (Coupling)
        transmission_df = pd.DataFrame({
            'Frequency_GHz': freq_ghz,
            'S21_dB': s_params_db[:, 1, 0],  # Port 1 to Port 2
            'S31_dB': s_params_db[:, 2, 0],  # Port 1 to Port 3
            'S32_dB': s_params_db[:, 2, 1]   # Port 2 to Port 3
        })

        # DataFrame 3: Isolation (negative of coupling)
        isolation_df = pd.DataFrame({
            'Frequency_GHz': freq_ghz,
            'Isolation_Ant1_2_dB': -s_params_db[:, 1, 0],  # Negative of S21
            'Isolation_Ant1_3_dB': -s_params_db[:, 2, 0],  # Negative of S31
            'Isolation_Ant2_3_dB': -s_params_db[:, 2, 1]   # Negative of S32
        })

        # DataFrame 4: Total Coupling to each antenna
        total_coupling_df = pd.DataFrame({
            'Frequency_GHz': freq_ghz,
            'Total_Coupling_Ant1_dB': results['total_coupling']['Antenna 1'],
            'Total_Coupling_Ant2_dB': results['total_coupling']['Antenna 2'],
            'Total_Coupling_Ant3_dB': results['total_coupling']['Antenna 3']
        })

        # DataFrame 5: VSWR Summary
        vswr_data = []
        for i in range(3):
            vswr_data.append({
                'Antenna': f'Antenna {i+1}',
                'VSWR_Min': results['vswr'][f'Antenna {i+1}']['min'],
                'VSWR_Max': results['vswr'][f'Antenna {i+1}']['max'],
                'VSWR_Center': results['vswr'][f'Antenna {i+1}']['center']
            })
        vswr_summary_df = pd.DataFrame(vswr_data)

        # DataFrame 6: Isolation Summary
        isolation_summary_data = []
        for iso_data in results['isolation']:
            isolation_summary_data.append({
                'Antenna_Pair': iso_data['pair'],
                'S_Parameter': iso_data['label'],
                'Min_Isolation_dB': iso_data['min_isolation'],
                'Avg_Isolation_dB': iso_data['avg_isolation'],
                'Worst_Freq_GHz': iso_data['worst_freq']
            })
        isolation_summary_df = pd.DataFrame(isolation_summary_data)

        # DataFrame 7: S-Parameter Matrix at Center Frequency
        center_idx = len(frequencies) // 2
        s_matrix_data = []
        for i in range(3):
            for j in range(3):
                s_matrix_data.append({
                    'Input_Port': f'Port {j+1}',
                    'Output_Port': f'Port {i+1}',
                    'S_Parameter': f'S{i+1}{j+1}',
                    'Magnitude_dB': s_params_db[center_idx, i, j],
                    'Frequency_GHz': freq_ghz[center_idx]
                })

16                                                                                                                        Altair SLC

        s_matrix_df = pd.DataFrame(s_matrix_data)

        return {
            'reflection': reflection_df,
            'transmission': transmission_df,
            'isolation': isolation_df,
            'total_coupling': total_coupling_df,
            'vswr_summary': vswr_summary_df,
            'isolation_summary': isolation_summary_df,
            's_matrix': s_matrix_df
        }

    def calculate_isolation(self, s_params_db):
        """Calculate isolation metrics from S-parameters"""
        # Minimum isolation (worst-case coupling)
        s21_min = np.min(s_params_db[:, 1, 0])
        s31_min = np.min(s_params_db[:, 2, 0])
        s32_min = np.min(s_params_db[:, 2, 1])

        # Average isolation in operating band
        center_idx = len(s_params_db) // 2
        band_idx = slice(center_idx - 50, center_idx + 50)

        avg_isolation = {
            'S21_avg': np.mean(s_params_db[band_idx, 1, 0]),
            'S31_avg': np.mean(s_params_db[band_idx, 2, 0]),
            'S32_avg': np.mean(s_params_db[band_idx, 2, 1])
        }

        return s21_min, s31_min, s32_min, avg_isolation

    def analyze_cosite_interference(self, frequencies, s_params_db):
        """Perform co-site interference analysis"""
        results = {}

        # 1. Reflection coefficient analysis (match quality)
        results['vswr'] = {}
        for i in range(3):
            s11 = s_params_db[:, i, i]
            # Calculate VSWR from S11
            gamma = 10**(s11/20)
            vswr = (1 + gamma) / (1 - gamma)
            results['vswr'][f'Antenna {i+1}'] = {
                'min': np.min(vswr),
                'max': np.max(vswr),
                'center': vswr[len(vswr)//2]
            }

        # 2. Isolation analysis
        isolation_data = []
        pairs = [(0, 1, 'S21'), (0, 2, 'S31'), (1, 2, 'S32')]

        for i, j, label in pairs:
            coupling = s_params_db[:, j, i]
            isolation = -coupling  # Isolation is negative of coupling in dB
            isolation_data.append({
                'pair': f'{i+1}-{j+1}',
                'label': label,
                'min_isolation': np.min(isolation),
                'avg_isolation': np.mean(isolation[50:-50]),
                'worst_freq': frequencies[np.argmin(isolation)] / 1e9
            })


17                                                                                                                        Altair SLC

        results['isolation'] = isolation_data

        # 3. Calculate total coupling to each antenna
        results['total_coupling'] = {}
        for i in range(3):
            total_power = 0
            for j in range(3):
                if i != j:
                    # Convert from dB to linear, sum power, convert back to dB
                    power_linear = 10**(s_params_db[:, j, i]/10)
                    total_power += power_linear
            results['total_coupling'][f'Antenna {i+1}'] = 10 * np.log10(total_power + 1e-10)

        return results

def plot_s_parameters(frequencies, s_params_db, results, dataframes=None):
    """Create comprehensive visualization of S-parameters"""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig)

    # Convert frequencies to GHz for plotting
    freq_ghz = frequencies / 1e9

    # Plot 1: Reflection coefficients (S11, S22, S33)
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(3):
        ax1.plot(freq_ghz, s_params_db[:, i, i],
                label=f'S{i+1}{i+1}', color=colors[i], linewidth=2)

    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Magnitude (dB)')
    ax1.set_title('Reflection Coefficients')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([-30, 0])

    # Plot 2: Transmission coefficients (coupling)
    ax2 = fig.add_subplot(gs[0, 1])
    coupling_pairs = [(0, 1, 'S21'), (0, 2, 'S31'), (1, 2, 'S32')]
    for i, j, label in coupling_pairs:
        ax2.plot(freq_ghz, s_params_db[:, j, i],
                label=label, linewidth=2)

    ax2.set_xlabel('Frequency (GHz)')
    ax2.set_ylabel('Magnitude (dB)')
    ax2.set_title('Transmission Coefficients (Coupling)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([-50, -10])

    # Plot 3: Isolation between antenna pairs
    ax3 = fig.add_subplot(gs[0, 2])
    for i, j, label in coupling_pairs:
        isolation = -s_params_db[:, j, i]  # Isolation is negative of coupling
        ax3.plot(freq_ghz, isolation,
                label=f'Ant {i+1}-{j+1}', linewidth=2)

    ax3.set_xlabel('Frequency (GHz)')
    ax3.set_ylabel('Isolation (dB)')
    ax3.set_title('Antenna-to-Antenna Isolation')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

18                                                                                                                        Altair SLC

    ax3.set_ylim([10, 50])

    # Plot 4: 3D visualization of coupling matrix at center frequency
    ax4 = fig.add_subplot(gs[1, :])
    center_idx = len(frequencies) // 2

    # Create coupling matrix heatmap
    coupling_matrix = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            coupling_matrix[i, j] = s_params_db[center_idx, j, i]

    im = ax4.imshow(coupling_matrix, cmap='viridis', aspect='auto')
    ax4.set_title(f'S-parameter Matrix at {freq_ghz[center_idx]:.2f} GHz')
    ax4.set_xticks([0, 1, 2])
    ax4.set_yticks([0, 1, 2])
    ax4.set_xticklabels(['Port 1', 'Port 2', 'Port 3'])
    ax4.set_yticklabels(['Port 1', 'Port 2', 'Port 3'])

    # Add text annotations
    for i in range(3):
        for j in range(3):
            text = ax4.text(j, i, f'{coupling_matrix[i, j]:.1f} dB',
                          ha="center", va="center",
                          color="white" if coupling_matrix[i, j] < -25 else "black")

    plt.colorbar(im, ax=ax4, label='Magnitude (dB)')

    # Plot 5: Total coupling to each antenna
    ax5 = fig.add_subplot(gs[2, 0])
    for i in range(3):
        total_coupling = results['total_coupling'][f'Antenna {i+1}']
        ax5.plot(freq_ghz, total_coupling,
                label=f'Antenna {i+1}', linewidth=2)

    ax5.set_xlabel('Frequency (GHz)')
    ax5.set_ylabel('Total Coupling (dB)')
    ax5.set_title('Total Interference Coupling per Antenna')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_ylim([-40, -10])

    # Plot 6: VSWR analysis
    ax6 = fig.add_subplot(gs[2, 1])
    antenna_labels = []
    vswr_center = []

    for i in range(3):
        antenna_labels.append(f'Ant {i+1}')
        vswr_center.append(results['vswr'][f'Antenna {i+1}']['center'])

    bars = ax6.bar(antenna_labels, vswr_center, color=colors)
    ax6.set_ylabel('VSWR at Center Frequency')
    ax6.set_title('Voltage Standing Wave Ratio')
    ax6.grid(True, alpha=0.3, axis='y')

    # Add VSWR threshold lines
    ax6.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='VSWR=2.0')
    ax6.axhline(y=1.5, color='green', linestyle='--', alpha=0.5, label='VSWR=1.5')
    ax6.legend()

    # Plot 7: Isolation summary
    ax7 = fig.add_subplot(gs[2, 2])

19                                                                                                                        Altair SLC

    isolation_pairs = []
    min_isolation_values = []

    for iso_data in results['isolation']:
        isolation_pairs.append(iso_data['pair'])
        min_isolation_values.append(iso_data['min_isolation'])

    bars = ax7.bar(isolation_pairs, min_isolation_values, color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
    ax7.set_ylabel('Minimum Isolation (dB)')
    ax7.set_title('Worst-case Antenna Isolation')
    ax7.grid(True, alpha=0.3, axis='y')

    # Add isolation requirement line
    ax7.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='20 dB Requirement')
    ax7.legend()

    # Add value labels on bars
    for bar, val in zip(bars, min_isolation_values):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f} dB', ha='center', va='bottom')

    plt.suptitle('Three-Antenna Co-site Interference Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()

    return fig

def generate_report(results, frequencies, s_params_db, dataframes):
    """Generate a text report of the analysis"""

    with open("d:/lgs/txt/lgs_summary.txt", 'a') as g:
        print("Starting Co-site Interference Analysis for Three-Antenna System...", file=g)
        print("="*70, file=g)
        print("CO-SITE INTERFERENCE ANALYSIS REPORT", file=g)
        print("="*70, file=g)

        print("\n1. REFLECTION COEFFICIENT SUMMARY (S11, S22, S33):", file=g)
        print("-"*50, file=g)
        for i in range(3):
            vswr_data = results['vswr'][f'Antenna {i+1}']
            print(f"Antenna {i+1} (S{i+1}{i+1}):", file=g)
            print(f"  - VSWR range: {vswr_data['min']:.2f} - {vswr_data['max']:.2f}", file=g)
            print(f"  - VSWR at center: {vswr_data['center']:.2f}", file=g)
            print(f"  - Match quality: {'Good' if vswr_data['center'] < 2.0 else 'Marginal' if vswr_data['center'] < 3.0 else 'Poor'}", file=g)

        print("\n2. ISOLATION ANALYSIS:", file=g)
        print("-"*50, file=g)
        for iso_data in results['isolation']:
            print(f"{iso_data['label']} (Antenna {iso_data['pair'][0]} to {iso_data['pair'][-1]}):", file=g)
            print(f"  - Minimum isolation: {iso_data['min_isolation']:.1f} dB", file=g)
            print(f"  - Average isolation: {iso_data['avg_isolation']:.1f} dB", file=g)
            print(f"  - Worst frequency: {iso_data['worst_freq']:.3f} GHz", file=g)
            print(f"  - Status: {'? PASS' if iso_data['min_isolation'] > 20 else '? MARGINAL' if iso_data['min_isolation'] > 15 else '? FAIL'}", file=g)

        print("\n3. DATAFRAME SUMMARIES:", file=g)
        print("-"*50, file=g)
        print("DataFrames created for frequency domain analysis:", file=g)
        for df_name, df in dataframes.items():
            print(f"  - {df_name}: {df.shape[0]} rows × {df.shape[1]} columns", file=g)

        print("\n4. INTERFERENCE ASSESSMENT:", file=g)
        print("-"*50, file=g)


20                                                                                                                        Altair SLC

        # Calculate worst-case scenarios
        center_idx = len(frequencies) // 2
        worst_coupling = np.min(s_params_db[:, :, :])
        worst_pair = np.unravel_index(np.argmin(s_params_db[:, :, :]), s_params_db.shape)

        print(f"Worst-case coupling: {worst_coupling:.1f} dB", file=g)
        print(f"Occurs at: {frequencies[worst_pair[0]]/1e9:.3f} GHz", file=g)
        print(f"Between: Port {worst_pair[2]+1} ? Port {worst_pair[1]+1}", file=g)

        print("\n5. RECOMMENDATIONS:", file=g)
        print("-"*50, file=g)

        # Generate recommendations based on analysis
        recommendations = []

        # Check isolation requirements
        isolation_status = all(iso['min_isolation'] > 20 for iso in results['isolation'])
        if not isolation_status:
            recommendations.append("- Consider increasing antenna spacing or adding shielding")
            recommendations.append("- Implement frequency separation or filtering")

        # Check VSWR
        vswr_status = all(v['center'] < 2.0 for v in results['vswr'].values())
        if not vswr_status:
            recommendations.append("- Optimize antenna matching networks")

        if not recommendations:
            recommendations.append("- System meets basic co-site requirements")
            recommendations.append("- Monitor performance with actual hardware")

        for rec in recommendations:
            print(rec, file=g)

        print("\n" + "="*70, file=g)

        # Save DataFrames to CSV files
        print("\nSaving DataFrames to CSV files...", file=g)
        for df_name, df in dataframes.items():
            filename = f"d:/lgs/csv/{df_name}_data.csv"
            df.to_csv(filename, index=False)
            print(f"  - Saved {df_name} to {filename}", file=g)

        print("\nSaving DataFrames to R rds files...", file=g)
        for df_name, df in dataframes.items():
            filename = f"d:/lgs/rds/{df_name}_data.rds"
            # Correct: Pass the DataFrame 'df', not the name '{df_name}'
            pr.write_rds(filename,df)  # Note: parameters swapped - DataFrame first, then filename
            print(f"  - Saved {df_name} to {filename}", file=g)

def main():
    """Main function to run the analysis"""

    with open("d:/lgs/txt/lgs_summary.txt", 'w') as h:

        # Create antenna system
        antenna_system = ThreeAntennaSystem(
            center_freq=2.4e9,  # 2.4 GHz center frequency
            bandwidth=500e6     # 500 MHz bandwidth
        )

        # Generate S-parameters
        frequencies, s_params_db = antenna_system.generate_s_parameters()


21                                                                                                                        Altair SLC

        # Perform analysis
        results = antenna_system.analyze_cosite_interference(frequencies, s_params_db)

        # Create pandas DataFrames for frequency domain data
        dataframes = antenna_system.create_frequency_dataframes(frequencies, s_params_db, results)

        # Display DataFrame information
        print("="*60, file=h)
        print("FREQUENCY DOMAIN DATAFRAMES CREATED:", file=h)
        print("="*60, file=h)

        for name, df in dataframes.items():
            print(f"\n{name.upper()} DataFrame:", file=h)
            print("-"*40, file=h)
            print(f"Shape: {df.shape}", file=h)
            print(f"Columns: {list(df.columns)}", file=h)
            if df.shape[0] > 5:
                print("\nFirst 5 rows:", file=h)
                print(df.head(), file=h)
            else:
                print(df, file=h)
            print("\nSummary statistics:", file=h)
            if 'Frequency_GHz' in df.columns and df.shape[0] > 3:
                print(df.describe(), file=h)

        # Generate report with DataFrames
        generate_report(results, frequencies, s_params_db, dataframes)

        # Create visualizations
        fig = plot_s_parameters(frequencies, s_params_db, results, dataframes)

        # Save results
        plt.savefig('d:/lgs/png/cosite_interference_analysis.png', dpi=150, bbox_inches='tight')
        print("\nAnalysis complete! Results saved to 'd:/lgs/png/cosite_interference_analysis.png'", file=h)

        # Display additional metrics
        print("\nAdditional Metrics:", file=h)
        print("-"*40, file=h)

        # Calculate and display isolation metrics
        s21_min, s31_min, s32_min, avg_isolation = antenna_system.calculate_isolation(s_params_db)

        print(f"Minimum S21 (Ant1?Ant2): {s21_min:.1f} dB", file=h)
        print(f"Minimum S31 (Ant1?Ant3): {s31_min:.1f} dB", file=h)
        print(f"Minimum S32 (Ant2?Ant3): {s32_min:.1f} dB", file=h)

        print(f"\nAverage Isolation in Operating Band:", file=h)
        print(f"S21 average: {avg_isolation['S21_avg']:.1f} dB", file=h)
        print(f"S31 average: {avg_isolation['S31_avg']:.1f} dB", file=h)
        print(f"S32 average: {avg_isolation['S32_avg']:.1f} dB", file=h)

        # Example: Access specific data from DataFrames
        print("\n" + "="*60, file=h)
        print("EXAMPLE DATA ACCESS FROM DATAFRAMES:", file=h)
        print("="*60, file=h)

        # Get center frequency data
        center_idx = len(frequencies) // 2
        center_freq_ghz = frequencies[center_idx] / 1e9

        print(f"\nData at center frequency ({center_freq_ghz:.2f} GHz):", file=h)
        print("-"*40, file=h)


22                                                                                                                        Altair SLC

        # Reflection coefficients at center frequency
        reflection_center = dataframes['reflection'].iloc[center_idx]
        print(f"Reflection Coefficients:", file=h)
        print(f"  S11: {reflection_center['S11_dB']:.1f} dB", file=h)
        print(f"  S22: {reflection_center['S22_dB']:.1f} dB", file=h)
        print(f"  S33: {reflection_center['S33_dB']:.1f} dB", file=h)

        # Transmission coefficients at center frequency
        transmission_center = dataframes['transmission'].iloc[center_idx]
        print(f"\nTransmission Coefficients:")
        print(f"  S21: {transmission_center['S21_dB']:.1f} dB", file=h)
        print(f"  S31: {transmission_center['S31_dB']:.1f} dB", file=h)
        print(f"  S32: {transmission_center['S32_dB']:.1f} dB", file=h)

        # Find worst isolation frequency
        isolation_df = dataframes['isolation']
        worst_isolation_idx = isolation_df[['Isolation_Ant1_2_dB', 'Isolation_Ant1_3_dB', 'Isolation_Ant2_3_dB']].min().idxmin()
        worst_value = isolation_df[worst_isolation_idx].min()
        worst_freq = isolation_df.loc[isolation_df[worst_isolation_idx].idxmin(), 'Frequency_GHz']

        print(f"\nWorst Isolation: {worst_value:.1f} dB at {worst_freq:.3f} GHz", file=h)
        print(f"Parameter: {worst_isolation_idx}", file=h)

        plt.show()

if __name__ == "__main__":
    main()

NOTE: 578 records were read from file 'c:\temp\py_pgm.py'
      The minimum record length was 80
      The maximum record length was 152
NOTE: 579 records were written to file 'c:\temp\py_pgm.pyx'
      The minimum record length was 16
      The maximum record length was 152
NOTE: The data step took :
      real time : 0.015
      cpu time  : 0.015



NOTE: The infile rut is:
      Unnamed Pipe Access Device,
      Process=d:\Python310\python.exe c:/temp/py_pgm.pyx 2> c:/temp/py_pgm.log,
      Lrecl=32767, Recfm=V


Transmission Coefficients:
NOTE: 2 records were written to file PRINT

NOTE: 2 records were read from file rut
      The minimum record length was 0
      The maximum record length was 26
NOTE: The data step took :
      real time : 8.934
      cpu time  : 0.000



NOTE: The infile 'c:\temp\py_pgm.log' is:
      Filename='c:\temp\py_pgm.log',
      Owner Name=T7610\Roger,
      File size (bytes)=0,
      Create Time=10:07:40 Dec 18 2025,

23                                                                                                                        Altair SLC

      Last Accessed=11:35:24 Dec 18 2025,
      Last Modified=11:35:24 Dec 18 2025,
      Lrecl=32767, Recfm=V

NOTE: No records were read from file 'c:\temp\py_pgm.log'
NOTE: The data step took :
      real time : 0.000
      cpu time  : 0.000


650
ERROR: Error printed on page 1

NOTE: Submitted statements took :
      real time : 9.108
      cpu time  : 0.093


/*___           _
 / _ \   _ __  | | ___   __ _
| (_) | | `__| | |/ _ \ / _` |
 \__, | | |    | | (_) | (_| |
   /_/  |_|    |_|\___/ \__, |
                        |___/
*/

1                                          Altair SLC     11:38 Thursday, December 18, 2025

NOTE: Copyright 2002-2025 World Programming, an Altair Company
NOTE: Altair SLC 2026 (05.26.01.00.000758)
      Licensed to Roger DeAngelis
NOTE: This session is executing on the X64_WIN11PRO platform and is running in 64 bit mode

NOTE: AUTOEXEC processing beginning; file is C:\wpsoto\autoexec.sas
NOTE: AUTOEXEC source line
1       +  ï»¿ods _all_ close;
           ^
ERROR: Expected a statement keyword : found "?"
NOTE: Library workx assigned as follows:
      Engine:        SAS7BDAT
      Physical Name: d:\wpswrkx


NOTE: 1 record was written to file PRINT

NOTE: The data step took :
      real time : 0.021
      cpu time  : 0.000


NOTE: AUTOEXEC processing completed

1
2         libname lgs sas7bdat "d:/lgs";
NOTE: Library lgs assigned as follows:
      Engine:        SAS7BDAT
      Physical Name: d:\lgs

3         options set=RHOME 'D:\r414';
4         proc r;
5         submit;
6         isolation     <-readRDS('d:/lgs/rds/isolation_data.rds')
7         reflection    <-readRDS('d:/lgs/rds/reflection_data.rds')
8         total_coupling<-readRDS('d:/lgs/rds/total_coupling_data.rds')
9         transmission  <-readRDS('d:/lgs/rds/transmission_data.rds')
10        endsubmit;
NOTE: Using R version 4.5.1 (2025-06-13 ucrt) from d:\r451

NOTE: Submitting statements to R:

> isolation     <-readRDS('d:/lgs/rds/isolation_data.rds')
> reflection    <-readRDS('d:/lgs/rds/reflection_data.rds')
> total_coupling<-readRDS('d:/lgs/rds/total_coupling_data.rds')
> transmission  <-readRDS('d:/lgs/rds/transmission_data.rds')

NOTE: Processing of R statements complete

11        import data=lgs.isolation       r=isolation     ;
NOTE: Creating data set 'LGS.isolation' from R data frame 'isolation'
NOTE: Column names modified during import of 'isolation'
NOTE: Data set "LGS.isolation" has 501 observation(s) and 4 variable(s)

12        import data=lgs.reflection      r=reflection    ;
NOTE: Creating data set 'LGS.reflection' from R data frame 'reflection'
NOTE: Column names modified during import of 'reflection'
NOTE: Data set "LGS.reflection" has 501 observation(s) and 4 variable(s)

13        import data=lgs.total_coupling  r=total_coupling;
NOTE: Creating data set 'LGS.total_coupling' from R data frame 'total_coupling'
NOTE: Column names modified during import of 'total_coupling'
NOTE: Data set "LGS.total_coupling" has 501 observation(s) and 4 variable(s)

2                                                                                                                         Altair SLC


14        import data=lgs.transmission    r=transmission  ;
NOTE: Creating data set 'LGS.transmission' from R data frame 'transmission'
NOTE: Column names modified during import of 'transmission'
NOTE: Data set "LGS.transmission" has 501 observation(s) and 4 variable(s)

15        ;quit;run;
NOTE: Procedure r step took :
      real time : 0.415
      cpu time  : 0.015


ERROR: Error printed on page 1

NOTE: Submitted statements took :
      real time : 0.491
      cpu time  : 0.062

/*              _
  ___ _ __   __| |
 / _ \ `_ \ / _` |
|  __/ | | | (_| |
 \___|_| |_|\__,_|

*/

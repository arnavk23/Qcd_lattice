"""
Master Demo Script - Run All In-Depth Demos

This script runs all the comprehensive in-depth demos for the QCD Lattice
Monte Carlo implementations:
1. Harmonic Oscillator In-Depth Demo
2. 1D Scalar Field Theory In-Depth Demo  
3. HMC In-Depth Demo
4. Metropolis Algorithm In-Depth Demo

All plots are saved to their respective directories under plots/
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

def run_demo(demo_script, demo_name):
    """Run a single demo script and capture output."""
    print(f"\\n{'=' * 80}")
    print(f"STARTING: {demo_name}")
    print(f"{'=' * 80}")
    
    try:
        # Run the demo script
        result = subprocess.run(
            [sys.executable, demo_script],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode == 0:
            print(f"✓ SUCCESS: {demo_name} completed successfully")
            if result.stdout:
                print("\\nOutput:")
                print(result.stdout)
        else:
            print(f"✗ ERROR: {demo_name} failed with return code {result.returncode}")
            if result.stderr:
                print("\\nError output:")
                print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT: {demo_name} exceeded 30 minute timeout")
        return False
    except Exception as e:
        print(f"✗ EXCEPTION: {demo_name} failed with exception: {e}")
        return False
    
    return True

def main():
    """Main function to run all demos."""
    print("QCD LATTICE MONTE CARLO - COMPREHENSIVE DEMO SUITE")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {Path.cwd()}")
    
    # Check if plot directories exist, create if needed
    plot_dirs = [
        "plots/harmonic_oscillator",
        "plots/field_theory", 
        "plots/hmc",
        "plots/metropolis"
    ]
    
    for plot_dir in plot_dirs:
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
    
    # Define demos to run
    demos = [
        ("demo_harmonic_oscillator_indepth.py", "Harmonic Oscillator In-Depth Demo"),
        ("demo_field_theory_indepth.py", "1D Scalar Field Theory In-Depth Demo"),
        ("demo_hmc_indepth.py", "HMC In-Depth Demo"),
        ("demo_metropolis_indepth.py", "Metropolis Algorithm In-Depth Demo")
    ]
    
    # Track results
    success_count = 0
    total_demos = len(demos)
    
    # Run each demo
    for demo_script, demo_name in demos:
        if run_demo(demo_script, demo_name):
            success_count += 1
        else:
            print(f"\\n⚠️  WARNING: {demo_name} failed, continuing with remaining demos...")
    
    # Final summary
    print(f"\\n{'=' * 80}")
    print("COMPREHENSIVE DEMO SUITE COMPLETED")
    print(f"{'=' * 80}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Success rate: {success_count}/{total_demos} demos completed successfully")
    
    if success_count == total_demos:
        print("✓ ALL DEMOS COMPLETED SUCCESSFULLY!")
    else:
        print(f"⚠️  {total_demos - success_count} demos failed")
    
    # List all generated plots
    print(f"\\nGenerated plots can be found in:")
    for plot_dir in plot_dirs:
        plot_path = Path(plot_dir)
        if plot_path.exists():
            png_files = list(plot_path.glob("*.png"))
            txt_files = list(plot_path.glob("*.txt"))
            if png_files or txt_files:
                print(f"  {plot_dir}/ ({len(png_files)} plots, {len(txt_files)} reports)")
    
    # Create master summary
    summary_path = Path("plots/master_demo_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("QCD LATTICE MONTE CARLO - COMPREHENSIVE DEMO SUITE SUMMARY\\n")
        f.write("=" * 65 + "\\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("DEMOS COMPLETED:\\n")
        f.write("-" * 20 + "\\n")
        for demo_script, demo_name in demos:
            status = "✓" if run_demo != False else "✗"
            f.write(f"{status} {demo_name}\\n")
        
        f.write(f"\\nSUCCESS RATE: {success_count}/{total_demos}\\n\\n")
        
        f.write("GENERATED CONTENT:\\n")
        f.write("-" * 20 + "\\n")
        for plot_dir in plot_dirs:
            plot_path = Path(plot_dir)
            if plot_path.exists():
                png_files = list(plot_path.glob("*.png"))
                txt_files = list(plot_path.glob("*.txt"))
                if png_files or txt_files:
                    f.write(f"{plot_dir}/: {len(png_files)} plots, {len(txt_files)} reports\\n")
        
        f.write(f"\\nTOTAL PLOTS: {sum(len(list(Path(d).glob('*.png'))) for d in plot_dirs if Path(d).exists())}\\n")
    
    print(f"\\nMaster summary saved to: {summary_path}")
    
    return success_count == total_demos

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Fix test2.hdf5 - Füge fehlende Metadaten hinzu
"""

import h5py
import json
import shutil
import os

def fix_hdf5_metadata(input_file, output_file=None, env_name="Isaac-Lift-Cube-Franka-IK-Abs-v0", no_backup=False):
    """
    Füge fehlende env_args und total Attribute zur HDF5 Datei hinzu.
    
    Args:
        input_file: Pfad zur Input-Datei
        output_file: Pfad zur Output-Datei (None = überschreibe Original)
        env_name: Name des Environments
        no_backup: Wenn True, kein Backup erstellen
    """
    # Backup erstellen
    if output_file is None:
        if not no_backup:
            backup_file = input_file + ".backup"
            print(f"📦 Erstelle Backup: {backup_file}")
            shutil.copy2(input_file, backup_file)
        output_file = input_file
    
    print(f"🔧 Öffne {input_file}...")
    
    # Datei im read-write Modus öffnen
    with h5py.File(input_file, 'r+') as f:
        if 'data' not in f:
            print("❌ Keine 'data' Gruppe gefunden!")
            return False
        
        data_group = f['data']
        print(f"✓ data Gruppe gefunden mit {len(data_group.keys())} demos")
        
        # Zähle total steps
        total_steps = 0
        for demo_name in data_group.keys():
            if demo_name.startswith('demo_'):
                demo = data_group[demo_name]
                if 'actions' in demo:
                    num_samples = len(demo['actions'])
                    total_steps += num_samples
                    # Setze num_samples Attribut falls nicht vorhanden
                    if 'num_samples' not in demo.attrs:
                        demo.attrs['num_samples'] = num_samples
        
        print(f"  Total steps berechnet: {total_steps}")
        
        # Setze env_args
        if 'env_args' not in data_group.attrs:
            env_args = {
                'env_name': env_name,
                'type': 2  # ManagerBasedRLEnv type
            }
            data_group.attrs['env_args'] = json.dumps(env_args)
            print(f"✓ env_args hinzugefügt: {env_args}")
        else:
            print("  env_args bereits vorhanden")
        
        # Setze total
        if 'total' not in data_group.attrs:
            data_group.attrs['total'] = total_steps
            print(f"✓ total hinzugefügt: {total_steps}")
        else:
            print(f"  total bereits vorhanden: {data_group.attrs['total']}")
    
    print(f"✅ Datei wurde aktualisiert: {output_file}")
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Repariere HDF5 Metadaten für replay_demos.py")
    parser.add_argument("--input", default="datasets/test2.hdf5", help="Input HDF5 Datei")
    parser.add_argument("--output", default=None, help="Output HDF5 Datei (None = überschreibe Input)")
    parser.add_argument("--env_name", default="Isaac-Lift-Cube-Franka-IK-Abs-v0", 
                        help="Environment Name")
    parser.add_argument("--no-backup", action="store_true", help="Kein Backup erstellen")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HDF5 METADATA FIXER")
    print("=" * 60)
    print()
    
    success = fix_hdf5_metadata(
        args.input, 
        args.output,
        args.env_name,
        args.no_backup
    )
    
    if success:
        print()
        print("🎉 Fertig! Die Datei sollte jetzt mit replay_demos.py funktionieren.")
        print()
        print("Test mit:")
        print(f"  ./isaaclab.sh -p scripts/tools/replay_demos.py \\")
        print(f"    --task {args.env_name} \\")
        print(f"    --dataset_file {args.output or args.input}")
    else:
        print()
        print("❌ Fehler beim Reparieren der Datei")

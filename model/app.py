from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pymatgen.core import Structure, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
import traceback
import os

# Model definitions
ELEMENTS = ['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Ti','Fe']
EL_TO_IDX = {e:i for i,e in enumerate(ELEMENTS)}
N_ELEM = len(ELEMENTS)
MAX_ATOMS = 64
LATENT_DIM = 128

# Element colors for visualization (CPK coloring)
ELEMENT_COLORS = {
    'H': '#FFFFFF', 'He': '#D9FFFF', 'Li': '#CC80FF', 'Be': '#C2FF00',
    'B': '#FFB5B5', 'C': '#909090', 'N': '#3050F8', 'O': '#FF0D0D',
    'F': '#90E050', 'Ne': '#B3E3F5', 'Na': '#AB5CF2', 'Mg': '#8AFF00',
    'Al': '#BFA6A6', 'Si': '#F0C8A0', 'P': '#FF8000', 'S': '#FFFF30',
    'Cl': '#1FF01F', 'Ar': '#80D1E3', 'K': '#8F40D4', 'Ca': '#3DFF00',
    'Ti': '#BFC2C7', 'Fe': '#E06633'
}

# Element radii (van der Waals radii in Angstroms)
ELEMENT_RADII = {
    'H': 1.20, 'He': 1.40, 'Li': 1.82, 'Be': 1.53, 'B': 1.92,
    'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47, 'Ne': 1.54,
    'Na': 2.27, 'Mg': 1.73, 'Al': 1.84, 'Si': 2.10, 'P': 1.80,
    'S': 1.80, 'Cl': 1.75, 'Ar': 1.88, 'K': 2.75, 'Ca': 2.31,
    'Ti': 2.15, 'Fe': 2.04
}

class SpaceGroupEmbedding(nn.Module):
    def __init__(self, max_sg=230, emb_dim=64):
        super().__init__()
        self.embed = nn.Embedding(max_sg+1, emb_dim)
    
    def forward(self, x):
        return self.embed(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sg_emb = SpaceGroupEmbedding()
        in_dim = 9 + MAX_ATOMS*3 + MAX_ATOMS*N_ELEM
        self.fc1 = nn.Linear(in_dim + 64, 512)
        self.fc_mu = nn.Linear(512, LATENT_DIM)
        self.fc_logvar = nn.Linear(512, LATENT_DIM)
    
    def forward(self, lattice, frac, species_oh, sg_idx):
        x = torch.cat([lattice.view(lattice.size(0), -1), 
                       frac.view(frac.size(0), -1), 
                       species_oh.view(species_oh.size(0), -1)], dim=-1)
        sg_e = self.sg_emb(sg_idx)
        x = torch.cat([x, sg_e], dim=-1)
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sg_emb = SpaceGroupEmbedding()
        out_dim = 9 + MAX_ATOMS*3 + MAX_ATOMS*N_ELEM
        self.fc1 = nn.Linear(LATENT_DIM + 64, 512)
        self.fc_out = nn.Linear(512, out_dim)
    
    def forward(self, z, sg_idx):
        sg_e = self.sg_emb(sg_idx)
        x = torch.cat([z, sg_e], dim=-1)
        h = F.relu(self.fc1(x))
        out = self.fc_out(h)
        lattice = out[:, :9].view(-1,3,3)
        frac = out[:,9:9+MAX_ATOMS*3].view(-1,MAX_ATOMS,3)
        species_logits = out[:,9+MAX_ATOMS*3:].view(-1,MAX_ATOMS,N_ELEM)
        return lattice, frac, species_logits

class CVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder()
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, lattice, frac, species_oh, sg_idx):
        mu, logvar = self.enc(lattice, frac, species_oh, sg_idx)
        z = self.reparameterize(mu, logvar)
        lattice_rec, frac_rec, species_logits = self.dec(z, sg_idx)
        return lattice_rec, frac_rec, species_logits, mu, logvar

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model
model = CVAE()

# Try multiple possible model paths
possible_paths = [
    "./checkpoints/cvae_latest.pt",
    "./model/checkpoints/cvae_latest.pt",
    "../checkpoints/cvae_latest.pt",
    "./cvae_latest.pt",
    "cvae_latest.pt"
]

model_loaded = False
for MODEL_PATH in possible_paths:
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.eval()
            print(f"✅ Model loaded successfully from: {MODEL_PATH}")
            model_loaded = True
            break
        except Exception as e:
            print(f"⚠️ Failed to load from {MODEL_PATH}: {e}")
            continue

if not model_loaded:
    print("="*60)
    print("⚠️  WARNING: Could not load model!")
    print("="*60)
    print("Searched in the following locations:")
    for path in possible_paths:
        print(f"  - {path}")
    print("\nPlease ensure your model file 'cvae_latest.pt' is in one of these locations")
    print("or update the MODEL_PATH in this script.")
    print("\nThe API will still run but generation will fail until model is loaded.")
    print("="*60)

def calculate_bonds(structure, max_bond_distance=3.0):
    """Calculate bonds between atoms based on distance"""
    bonds = []
    atoms = structure.sites
    
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            distance = atoms[i].distance(atoms[j])
            if distance < max_bond_distance:
                bonds.append({
                    'atom1': i,
                    'atom2': j,
                    'distance': float(distance)
                })
    
    return bonds

def structure_to_xyz(structure):
    """Convert structure to XYZ format"""
    try:
        atoms = AseAtomsAdaptor.get_atoms(structure)
        xyz_str = f"{len(atoms)}\n\n"
        for atom in atoms:
            xyz_str += f"{atom.symbol} {atom.position[0]:.6f} {atom.position[1]:.6f} {atom.position[2]:.6f}\n"
        return xyz_str
    except Exception as e:
        print(f"Error converting to XYZ: {e}")
        return None

def generate_structure(spacegroup_idx, comp_dict, num_atoms=8, temperature=1.0):
    """Generate crystal structure using the CVAE model"""
    if not model_loaded:
        raise Exception("Model not loaded. Please check model path and restart the server.")
    
    try:
        # Ensure num_atoms is an integer
        num_atoms = int(num_atoms)
        
        z = torch.randn((1, LATENT_DIM)) * temperature
        sg_idx = torch.tensor([spacegroup_idx], dtype=torch.long)
        
        with torch.no_grad():
            lat_pred, frac_pred, species_logits = model.dec(z, sg_idx)
        
        lat = lat_pred.detach().numpy()[0]
        frac = frac_pred.detach().numpy()[0]
        
        # Use elements from composition
        elements_from_comp = list(comp_dict.keys())
        num_elements = len(elements_from_comp)
        species_symbols = [elements_from_comp[i % num_elements] for i in range(num_atoms)]
        
        # Build pymatgen Structure
        lattice = Lattice(lat)
        sites = []
        
        for i in range(num_atoms):
            species_symbol = species_symbols[i]
            if species_symbol in ELEMENTS:
                sites.append({
                    'species': [{"element": species_symbol, "occu": 1}],
                    'abc': frac[i].tolist()
                })
        
        if not sites:
            raise ValueError("No valid sites generated")
        
        structure = Structure.from_dict({
            'lattice': lattice.as_dict(),
            'sites': sites,
            'charge': 0
        })
        
        return structure
    except Exception as e:
        raise Exception(f"Structure generation failed: {str(e)}")

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "model_loaded": model_loaded,
        "message": "Flask API is running" if model_loaded else "Model not loaded"
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        
        # Extract parameters - FIXED: Handle composition as dict
        spacegroup = int(data.get('spacegroup', 225))
        composition_data = data.get('composition', {'Fe': 1, 'O': 1})
        num_atoms = int(data.get('num_atoms', 8))
        temperature = float(data.get('temperature', 1.0))
        
        # Validate composition is a dictionary
        if not isinstance(composition_data, dict):
            return jsonify({
                'success': False,
                'error': 'Composition must be a dictionary of element: amount pairs'
            }), 400
        
        # Convert composition values to float and validate elements
        comp_dict = {}
        for element, amount in composition_data.items():
            if element not in ELEMENTS:
                return jsonify({
                    'success': False,
                    'error': f'Invalid element: {element}. Must be one of: {", ".join(ELEMENTS)}'
                }), 400
            comp_dict[element] = float(amount)
        
        if not comp_dict:
            return jsonify({
                'success': False,
                'error': 'Composition cannot be empty'
            }), 400
        
        # Create composition string for logging
        composition_str = ''.join([f"{el}{int(amt) if amt == int(amt) else amt}" 
                                   for el, amt in comp_dict.items()])
        
        print(f"📝 Generating: {composition_str}, SG={spacegroup}, Atoms={num_atoms}, T={temperature}")
        
        # Validate inputs
        if not 1 <= spacegroup <= 230:
            return jsonify({
                'success': False,
                'error': 'Space group must be between 1 and 230'
            }), 400
        
        if not 1 <= num_atoms <= MAX_ATOMS:
            return jsonify({
                'success': False,
                'error': f'Number of atoms must be between 1 and {MAX_ATOMS}'
            }), 400
        
        # Generate structure
        structure = generate_structure(spacegroup, comp_dict, num_atoms, temperature)
        
        # Calculate properties
        lattice_params = structure.lattice
        volume = float(lattice_params.volume)
        
        # Calculate density (approximate)
        atomic_masses = {
            'H': 1, 'He': 4, 'Li': 7, 'Be': 9, 'B': 11, 'C': 12, 'N': 14, 'O': 16, 
            'F': 19, 'Ne': 20, 'Na': 23, 'Mg': 24, 'Al': 27, 'Si': 28, 'P': 31, 
            'S': 32, 'Cl': 35, 'Ar': 40, 'K': 39, 'Ca': 40, 'Ti': 48, 'Fe': 56
        }
        
        total_mass = sum(
            structure.composition.get_el_amt_dict().get(el, 0) * atomic_masses.get(el, 50)
            for el in comp_dict.keys()
        )
        density = (total_mass * 1.66054) / volume if volume > 0 else 0
        
        # Prepare atoms data with both cartesian and fractional coordinates
        atoms_data = []
        for i, site in enumerate(structure.sites):
            element = site.specie.symbol
            atoms_data.append({
                'element': element,
                'position': site.coords.tolist(),  # Cartesian coordinates
                'frac_coords': site.frac_coords.tolist()  # Fractional coordinates
            })
        
        # Calculate bonds
        bonds = calculate_bonds(structure)
        
        # Generate CIF and XYZ
        cif_data = structure.to(fmt="cif")
        xyz_data = structure_to_xyz(structure)
        
        # Prepare response matching frontend interface
        response = {
            'success': True,
            'formula': structure.formula,
            'spacegroup': spacegroup,
            'lattice_parameters': {
                'a': float(lattice_params.a),
                'b': float(lattice_params.b),
                'c': float(lattice_params.c),
                'alpha': float(lattice_params.alpha),
                'beta': float(lattice_params.beta),
                'gamma': float(lattice_params.gamma),
                'volume': volume
            },
            'atoms': atoms_data,
            'xyz_data': xyz_data,
            'cif_data': cif_data,
            'properties': {
                'volume': volume,
                'density': density,
                'num_atoms': len(structure.sites)
            }
        }
        
        print(f"✅ Generated: {structure.formula} with {len(structure.sites)} atoms")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error in generate endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/elements', methods=['GET'])
def get_elements():
    return jsonify({'elements': ELEMENTS})

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Crystal Structure Generator API',
        'version': '1.0',
        'endpoints': {
            'health': '/api/health',
            'elements': '/api/elements',
            'generate': '/api/generate (POST)'
        }
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Flask API Server")
    print("="*60)
    print(f"📍 Server: http://localhost:5000")
    print(f"🔍 Health: http://localhost:5000/api/health")
    print(f"📖 Elements: http://localhost:5000/api/elements")
    print(f"⚡ Generate: http://localhost:5000/api/generate (POST)")
    print("="*60)
    print(f"📦 Model Status: {'✅ Loaded' if model_loaded else '❌ Not Loaded'}")
    print("="*60)
    print("\n💡 Make sure your frontend API_URL is set to: http://localhost:5000/api")
    print("⚠️  Press CTRL+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
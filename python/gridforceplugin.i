%module gridforceplugin

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"

/*
 * The following lines are needed to handle std::vector
*/
%include "std_string.i"
%include "std_iostream.i"
%include "std_map.i"
%include "std_pair.i"
%include "std_set.i"
%include "std_vector.i"
%include "std_shared_ptr.i"
namespace std {
  %template(pairii) pair<int,int>;
  %template(vectord) vector<double>;
  %template(vectorddd) vector< vector< vector<double> > >;
  %template(vectori) vector<int>;
  %template(vectorii) vector < vector<int> >;
  %template(vectorpairii) vector< pair<int,int> >;
  %template(vectorstring) vector<string>;
  %template(mapstringstring) map<string,string>;
  %template(mapstringdouble) map<string,double>;
  %template(mapii) map<int,int>;
  %template(seti) set<int>;
}

%{
#include "GridForceTypes.h"
#include "GridData.h"
#include "GridForce.h"
#include "GridForceKernels.h"
#include "CachedGridData.h"
#include "IsolatedNonbondedForce.h"
#include "IsolatedNonbondedForceKernels.h"
#include "BondedHessian.h"
#include "NewtonMinimizer.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%feature("autodoc", "1");
%nodefaultctor;

// Exception handling for OpenMMException
%exception {
    try {
        $action
    } catch (const OpenMM::OpenMMException& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return NULL;
    }
}

using namespace OpenMM;

// Declare shared_ptr support for GridData (must be outside namespace)
%shared_ptr(GridForcePlugin::GridData)

%pythoncode %{
def _openmm_GridForce_director_call(force):
    """Helper to downcast Force* to GridForce* when retrieved from System"""
    if force is None:
        return None
    # Try to create a GridForce wrapper if the C++ object is actually a GridForce
    try:
        gf = GridForce.__new__(GridForce)
        gf.this = force.this
        gf.thisown = 0
        # Test if it's actually a GridForce by trying to call a GridForce-specific method
        _ = gf.getAutoGenerateGrid()
        return gf
    except:
        return force
%}

namespace GridForcePlugin {

enum class InvPowerMode {
    NONE = 0,
    RUNTIME = 1,
    STORED = 2
};

struct HessianAnalysis {
    std::vector<double> eigenvalues;
    std::vector<double> eigenvectors;
    std::vector<double> meanCurvature;
    std::vector<double> totalCurvature;
    std::vector<double> gaussianCurvature;
    std::vector<double> fracAnisotropy;
    std::vector<double> entropy;
    std::vector<double> minEigenvalue;
    std::vector<int> numNegative;
    double totalEntropy;
};

struct ParticleGroup {
    ParticleGroup(const std::string& name,
                  const std::vector<int>& particleIndices,
                  const std::vector<double>& scalingFactors = std::vector<double>());

    std::string name;
    std::vector<int> particleIndices;
    std::vector<double> scalingFactors;
};

class GridData {
public:
    GridData();
    GridData(int nx, int ny, int nz, double dx, double dy, double dz);

    static std::shared_ptr<GridData> loadFromFile(const std::string& filename);
    void saveToFile(const std::string& filename) const;

    // Dimension accessors
    int getNx() const;
    int getNy() const;
    int getNz() const;
    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getCounts(int& nx, int& ny, int& nz) const;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;

    // Spacing accessors
    double getDx() const;
    double getDy() const;
    double getDz() const;
    %apply double& OUTPUT {double& dx};
    %apply double& OUTPUT {double& dy};
    %apply double& OUTPUT {double& dz};
    void getSpacing(double& dx, double& dy, double& dz) const;
    %clear double& dx;
    %clear double& dy;
    %clear double& dz;

    // Origin accessors
    %apply double& OUTPUT {double& ox};
    %apply double& OUTPUT {double& oy};
    %apply double& OUTPUT {double& oz};
    void getOrigin(double& ox, double& oy, double& oz) const;
    %clear double& ox;
    %clear double& oy;
    %clear double& oz;
    void setOrigin(double x, double y, double z);

    // Data accessors
    const std::vector<double>& getValues() const;
    const std::vector<double>& getDerivatives() const;
    bool hasDerivatives() const;

    // Metadata accessors
    const std::string& getGridType() const;
    void setGridType(const std::string& type);
    double getInvPower() const;

    // Setters for construction
    void setValues(const std::vector<double>& vals);
    void setDerivatives(const std::vector<double>& derivs);
};

class GridForce : public OpenMM::Force {
public:
    GridForce();
    GridForce(std::shared_ptr<GridData> gridData);

    void setGridData(std::shared_ptr<GridData> gridData);
    std::shared_ptr<GridData> getGridData() const;

    void addGridCounts (int nx, int ny, int nz);
    void addGridSpacing (double dx, double dy, double dz);
    void addGridValue (double val);
    void setGridValues(const std::vector<double>& vals);
    const std::vector<double>& getGridValues() const;
    void addScalingFactor (double val);
    void setScalingFactor (int index, double val);

    void setAutoCalculateScalingFactors(bool enable);
    bool getAutoCalculateScalingFactors() const;
    void setScalingProperty(const std::string& property);
    const std::string& getScalingProperty() const;

    void setInvPowerMode(InvPowerMode mode, double inv_power);
    InvPowerMode getInvPowerMode() const;
    void applyInvPowerTransformation();

    double getInvPower() const;
    void setGridCap(double uMax);
    double getGridCap() const;
    void setOutOfBoundsRestraint(double k);
    double getOutOfBoundsRestraint() const;
    void setInterpolationMethod(int method);
    int getInterpolationMethod() const;

    void setAutoGenerateGrid(bool enable);
    bool getAutoGenerateGrid() const;
    void setGridType(const std::string& type);
    const std::string& getGridType() const;

    void setGridOrigin(double x, double y, double z);
    void getGridOrigin(double& OUTPUT, double& OUTPUT, double& OUTPUT) const;

    void setComputeDerivatives(bool compute);
    bool getComputeDerivatives() const;
    bool hasDerivatives() const;
    const std::vector<double>& getDerivatives() const;

    void setReceptorAtoms(const std::vector<int>& atomIndices);
    const std::vector<int>& getReceptorAtoms() const;
    void setLigandAtoms(const std::vector<int>& atomIndices);
    const std::vector<int>& getLigandAtoms() const;

    void setParticles(const std::vector<int>& particles);
    const std::vector<int>& getParticles() const;

    // Particle group management for multi-ligand workflows
    int addParticleGroup(const std::string& name,
                         const std::vector<int>& particleIndices,
                         const std::vector<double>& scalingFactors = std::vector<double>());
    int getNumParticleGroups() const;
    const ParticleGroup& getParticleGroup(int index) const;
    const ParticleGroup* getParticleGroupByName(const std::string& name) const;
    void removeParticleGroup(int index);
    void clearParticleGroups();

    std::vector<double> getParticleGroupEnergies(OpenMM::Context& context) const;
    std::vector<double> getParticleAtomEnergies(OpenMM::Context& context) const;
    std::vector<int> getParticleOutOfBoundsFlags(OpenMM::Context& context) const;

    // Hessian (second derivative) computation for normal modes analysis
    void computeHessian(OpenMM::Context& context) const;
    std::vector<double> getHessianBlocks(OpenMM::Context& context) const;

    // Hessian analysis for eigenvalues, curvature metrics, and entropy
    HessianAnalysis analyzeHessian(OpenMM::Context& context, float temperature = 300.0f) const;

    // Tiled grid streaming mode
    void setTiledMode(bool enable, int tileSize = 64, int memoryBudgetMB = 2048);
    bool getTiledMode() const;
    int getTileSize() const;
    int getMemoryBudgetMB() const;

    // Tiled file I/O (for generating/loading large grids tile-by-tile)
    void setTiledOutputFile(const std::string& filename, int tileSize = 32);
    const std::string& getTiledOutputFile() const;
    int getTiledOutputTileSize() const;
    void setTiledInputFile(const std::string& filename);
    const std::string& getTiledInputFile() const;

    void clearGridData();

    void setReceptorPositions(const std::vector<OpenMM::Vec3>& positions);
    void setReceptorPositionsFromArrays(const std::vector<double>& x,
                                        const std::vector<double>& y,
                                        const std::vector<double>& z);
    const std::vector<OpenMM::Vec3>& getReceptorPositions() const;

    %pythoncode %{
    def setReceptorPositionsFromLists(self, positions_list):
        """
        Set receptor positions from a list of 3-tuples or array-like positions.

        Args:
            positions_list: List of (x,y,z) tuples/lists in nanometers, or array with shape (N,3)
        """
        import numpy as np
        # Convert to numpy array for easy manipulation
        pos_array = np.asarray(positions_list, dtype=np.float64)
        if pos_array.ndim != 2 or pos_array.shape[1] != 3:
            raise ValueError("positions_list must be an Nx3 array or list of (x,y,z) tuples")

        # Extract x, y, z columns and call C++ method
        x = pos_array[:, 0].tolist()
        y = pos_array[:, 1].tolist()
        z = pos_array[:, 2].tolist()
        self.setReceptorPositionsFromArrays(x, y, z)

    def getHessianMatrices(self, context):
        """
        Compute and return Hessian blocks as (N, 3, 3) numpy array.

        This is a convenience wrapper around computeHessian() and getHessianBlocks()
        that returns the Hessian in matrix form suitable for normal modes analysis.

        For grid-based potentials, the Hessian is block-diagonal - each atom only
        contributes to its own 3x3 block since atoms interact independently with the grid.

        Args:
            context: OpenMM Context that has been evaluated (getState with forces)

        Returns:
            numpy.ndarray: Shape (N, 3, 3) array of Hessian blocks per atom.
                           Each block contains second derivatives:
                           [[d²V/dx², d²V/dxdy, d²V/dxdz],
                            [d²V/dydx, d²V/dy², d²V/dydz],
                            [d²V/dzdx, d²V/dzdy, d²V/dz²]]
                           Units are kJ/(mol·nm²).

        Raises:
            RuntimeError: If interpolation method doesn't support Hessian computation
                          (only bspline and triquintic are supported).

        Example:
            >>> # After minimization
            >>> gridforce.computeHessian(context)
            >>> H_blocks = gridforce.getHessianMatrices(context)
            >>> # H_blocks[i] is the 3x3 Hessian for atom i
        """
        import numpy as np

        # Compute Hessian on GPU
        self.computeHessian(context)

        # Get flat array: [dxx0, dyy0, dzz0, dxy0, dxz0, dyz0, dxx1, ...]
        flat = np.array(self.getHessianBlocks(context))

        if len(flat) == 0:
            return np.zeros((0, 3, 3))

        # Reshape to (N, 6)
        n_atoms = len(flat) // 6
        blocks = flat.reshape(n_atoms, 6)

        # Convert to (N, 3, 3) symmetric matrices
        # Layout: [dxx, dyy, dzz, dxy, dxz, dyz]
        H = np.zeros((n_atoms, 3, 3))
        H[:, 0, 0] = blocks[:, 0]  # dxx
        H[:, 1, 1] = blocks[:, 1]  # dyy
        H[:, 2, 2] = blocks[:, 2]  # dzz
        H[:, 0, 1] = H[:, 1, 0] = blocks[:, 3]  # dxy
        H[:, 0, 2] = H[:, 2, 0] = blocks[:, 4]  # dxz
        H[:, 1, 2] = H[:, 2, 1] = blocks[:, 5]  # dyz

        return H

    def getHessianAnalysis(self, context, temperature=300.0):
        """
        Compute and return comprehensive Hessian analysis with numpy arrays.

        This performs eigendecomposition of each 3x3 Hessian block using Cardano's
        analytical method, then computes derived metrics useful for binding site
        analysis and normal modes approximations.

        The analysis includes:
        - Eigenvalues (sorted ascending) for each atom
        - Eigenvectors (optional) for each atom
        - Curvature metrics: mean, total (trace), Gaussian (product)
        - Fractional anisotropy: 0=isotropic potential, 1=linear/directional
        - Per-atom harmonic entropy in kB units
        - Saddle point detection via negative eigenvalue count

        Args:
            context: OpenMM Context that has been evaluated (getState with forces)
            temperature: Temperature in Kelvin for entropy calculation (default: 300.0)

        Returns:
            dict: Dictionary with keys:
                'eigenvalues': numpy.ndarray (N, 3) - sorted ascending per atom
                'eigenvectors': numpy.ndarray (N, 3, 3) - one 3x3 matrix per atom
                'mean_curvature': numpy.ndarray (N,) - (λ1 + λ2 + λ3) / 3
                'total_curvature': numpy.ndarray (N,) - λ1 + λ2 + λ3 (trace of Hessian)
                'gaussian_curvature': numpy.ndarray (N,) - λ1 * λ2 * λ3
                'frac_anisotropy': numpy.ndarray (N,) - range [0,1]
                'entropy': numpy.ndarray (N,) - per-atom entropy in kB units (NaN at saddle points)
                'min_eigenvalue': numpy.ndarray (N,) - smallest eigenvalue per atom
                'num_negative': numpy.ndarray (N,) int - count of negative eigenvalues (0-3)
                'total_entropy': float - sum of per-atom entropies (excluding NaN)

        Raises:
            RuntimeError: If interpolation method doesn't support Hessian computation
                          (only bspline and triquintic are supported).

        Example:
            >>> # After minimization or energy evaluation
            >>> state = context.getState(getEnergy=True, getForces=True)
            >>> analysis = gridforce.getHessianAnalysis(context, temperature=300.0)
            >>>
            >>> # Find atoms at saddle points (not at true minimum)
            >>> saddle_atoms = np.where(analysis['num_negative'] > 0)[0]
            >>>
            >>> # Get average fractional anisotropy (measure of potential directionality)
            >>> avg_fa = np.mean(analysis['frac_anisotropy'])
            >>>
            >>> # Total configurational entropy contribution from grid
            >>> S_config = analysis['total_entropy']
        """
        import numpy as np

        # Call C++ analysis method
        result = self.analyzeHessian(context, temperature)

        n_atoms = len(result.meanCurvature)

        # Convert to numpy arrays with proper shapes
        analysis = {
            'eigenvalues': np.array(result.eigenvalues).reshape(n_atoms, 3) if n_atoms > 0 else np.zeros((0, 3)),
            'eigenvectors': np.array(result.eigenvectors).reshape(n_atoms, 3, 3) if n_atoms > 0 else np.zeros((0, 3, 3)),
            'mean_curvature': np.array(result.meanCurvature),
            'total_curvature': np.array(result.totalCurvature),
            'gaussian_curvature': np.array(result.gaussianCurvature),
            'frac_anisotropy': np.array(result.fracAnisotropy),
            'entropy': np.array(result.entropy),
            'min_eigenvalue': np.array(result.minEigenvalue),
            'num_negative': np.array(result.numNegative),
            'total_entropy': result.totalEntropy
        }

        return analysis
    %}

    void loadFromFile(const std::string& filename);
    void saveToFile(const std::string& filename) const;

    %apply std::vector<int> & OUTPUT { std::vector<int> & counts };
    %apply std::vector<double> & OUTPUT { std::vector<double> & spacing };
    %apply std::vector<double> & OUTPUT { std::vector<double> & vals };
    %apply std::vector<double> & OUTPUT { std::vector<double> & scaling_factors };
    void getGridParameters(std::vector<int>& counts, std::vector<double>& spacing, std::vector<double>& vals,
                           std::vector<double> &scaling_factors) const;
    %clear std::vector<int> & counts;
    %clear std::vector<double> & spacing;
    %clear std::vector<double> & vals;
    %clear std::vector<double> & scaling_factors;

    void updateParametersInContext(Context &context);
};

class IsolatedNonbondedForce : public OpenMM::Force {
public:
    IsolatedNonbondedForce();

    int getNumAtoms() const;
    void setNumAtoms(int numAtoms);

    void setParticles(const std::vector<int>& particles);
    const std::vector<int>& getParticles() const;

    void setAtomParameters(int index, double charge, double sigma, double epsilon);

    %apply double& OUTPUT {double& charge};
    %apply double& OUTPUT {double& sigma};
    %apply double& OUTPUT {double& epsilon};
    void getAtomParameters(int index, double& charge, double& sigma, double& epsilon) const;
    %clear double& charge;
    %clear double& sigma;
    %clear double& epsilon;

    void addExclusion(int atom1, int atom2);
    int getNumExclusions() const;

    %apply int& OUTPUT {int& atom1};
    %apply int& OUTPUT {int& atom2};
    void getExclusion(int index, int& atom1, int& atom2) const;
    %clear int& atom1;
    %clear int& atom2;

    int addException(int atom1, int atom2, double chargeProd, double sigma, double epsilon);
    int getNumExceptions() const;

    %apply int& OUTPUT {int& atom1_ex};
    %apply int& OUTPUT {int& atom2_ex};
    %apply double& OUTPUT {double& chargeProd};
    %apply double& OUTPUT {double& sigma_ex};
    %apply double& OUTPUT {double& epsilon_ex};
    void getExceptionParameters(int index, int& atom1_ex, int& atom2_ex, double& chargeProd,
                                 double& sigma_ex, double& epsilon_ex) const;
    %clear int& atom1_ex;
    %clear int& atom2_ex;
    %clear double& chargeProd;
    %clear double& sigma_ex;
    %clear double& epsilon_ex;

    void updateParametersInContext(Context &context);

    std::vector<double> computeHessian(OpenMM::Context& context);

    %pythoncode %{
    def getHessianMatrix(self, context):
        """
        Compute and return the full Hessian matrix as a numpy array.

        This computes the analytical Hessian (second derivatives) of the
        isolated nonbonded potential with respect to all atomic coordinates.

        Args:
            context: OpenMM Context containing current positions

        Returns:
            numpy.ndarray: Shape (3N, 3N) Hessian matrix where N is the number
                           of atoms. Units are kJ/(mol·nm²).

        Example:
            >>> H = isolated_nb_force.getHessianMatrix(context)
            >>> eigenvalues = np.linalg.eigvalsh(H)
        """
        import numpy as np
        flat = np.array(self.computeHessian(context))
        n = self.getNumAtoms()
        return flat.reshape(3*n, 3*n)
    %}
};

/**
 * BondedHessian computes analytical Hessians for bonded forces (CPU version).
 */
class BondedHessian {
public:
    BondedHessian();
    ~BondedHessian();

    void initialize(const OpenMM::System& system, OpenMM::Context& context);
    std::vector<double> computeHessian(OpenMM::Context& context);
    int getNumBonds() const;
    int getNumAngles() const;
    int getNumTorsions() const;

    %pythoncode %{
    def getHessianMatrix(self, context):
        """
        Compute and return the full Hessian matrix as a numpy array (CPU).

        This computes the analytical Hessian (second derivatives) of all
        bonded forces (HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce)
        with respect to all atomic coordinates.

        Args:
            context: OpenMM Context containing current positions

        Returns:
            numpy.ndarray: Shape (3N, 3N) Hessian matrix where N is the number
                           of particles in the System. Units are kJ/(mol·nm²).

        Example:
            >>> hessian_calc = BondedHessian()
            >>> hessian_calc.initialize(system, context)
            >>> H = hessian_calc.getHessianMatrix(context)
            >>> eigenvalues = np.linalg.eigvalsh(H)
        """
        import numpy as np
        flat = np.array(self.computeHessian(context))
        n = int(np.sqrt(len(flat)))
        return flat.reshape(n, n)
    %}
};

/**
 * NewtonMinimizer performs energy minimization using analytical Hessians.
 */
class NewtonMinimizer {
public:
    NewtonMinimizer();
    ~NewtonMinimizer();

    bool minimize(OpenMM::Context& context, double tolerance = 1.0, int maxIterations = 100);
    bool minimizeBondedOnly(OpenMM::Context& context, double tolerance = 10.0, int maxIterations = 50);
    int getNumIterations() const;
    double getFinalRMSForce() const;
    void setDamping(double lambda);
    void setLineSearch(bool enable);

    %pythoncode %{
    def minimizeToTolerance(self, context, force_tolerance=10.0, max_iterations=100):
        """
        Minimize energy until RMS force is below tolerance.

        This uses Newton-Raphson optimization with analytical Hessians,
        which provides quadratic convergence near minima. Much faster than
        gradient-based methods for small molecules.

        Args:
            context: OpenMM Context to minimize
            force_tolerance: RMS force tolerance in kJ/(mol·nm) (default: 10.0)
            max_iterations: Maximum Newton iterations (default: 100)

        Returns:
            dict: {'converged': bool, 'iterations': int, 'rms_force': float}

        Example:
            >>> minimizer = NewtonMinimizer()
            >>> result = minimizer.minimizeToTolerance(context, force_tolerance=1.0)
            >>> print(f"Converged in {result['iterations']} iterations")
        """
        converged = self.minimize(context, force_tolerance, max_iterations)
        return {
            'converged': converged,
            'iterations': self.getNumIterations(),
            'rms_force': self.getFinalRMSForce()
        }
    %}
};

class CalcGridForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {return "CalcGridForce";}
};

class CalcIsolatedNonbondedForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {return "CalcIsolatedNonbondedForce";}
};

} // namespace

%pythoncode %{
def castToGridForce(force):
    """
    Cast a generic Force object to GridForce if it's actually a GridForce.

    This is needed because when retrieving forces from a System via getForce(),
    they come back as generic Force objects even if they're actually GridForce objects.

    Usage:
        force = system.getForce(i)
        gridforce = gridforceplugin.castToGridForce(force)
        if gridforce is not None:
            gridforce.saveToFile("grid.grid")

    Returns GridForce if successful, None otherwise.
    """
    return _openmm_GridForce_director_call(force)

# when we import * from the python module, we only want to import the
# actual classes, and not the swigregistration methods, which have already
# been called, and are now unneeded by the user code, and only pollute the
# namespace
__all__ = [k for k in locals().keys() if not (k.endswith('_swigregister') or k.startswith('_'))]

def clearGridCache():
    """
    Clear the global grid data cache to free host memory.

    Call this after processing each system in batch workflows to prevent
    memory accumulation from cached grid data.
    """
    _gridforceplugin.clearGridCache()
%}

// Expose cache clearing function
%inline %{
void clearGridCache() {
    GridForcePlugin::GridDataCache::clearAll();
}
%}

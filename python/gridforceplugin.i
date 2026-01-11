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

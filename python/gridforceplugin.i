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
#include "GridForce.h"
#include "GridForceKernels.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
%}

%feature("autodoc", "1");
%nodefaultctor;

using namespace OpenMM;

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

class GridForce : public OpenMM::Force {
public:
    GridForce();

    void addGridCounts (int nx, int ny, int nz);
    void addGridSpacing (double dx, double dy, double dz);
    void addGridValue (double val);
    void addScalingFactor (double val);
    void setScalingFactor (int index, double val);

    void setAutoCalculateScalingFactors(bool enable);
    bool getAutoCalculateScalingFactors() const;
    void setScalingProperty(const std::string& property);
    const std::string& getScalingProperty() const;

    void setInvPower(double inv_power);
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

class CalcGridForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {return "CalcGridForce";}
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
%}

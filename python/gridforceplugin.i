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
  %template(vectorf) vector<float>;
  %template(vectorddd) vector< vector< vector<double> > >;
  %template(vectori) vector<int>;
  %template(vectorii) vector < vector<int> >;
  %template(vectorpairii) vector< pair<int,int> >;
  %template(vectorstring) vector<string>;
  %template(vectorll) vector<long long>;
  %template(mapstringstring) map<string,string>;
  %template(mapstringdouble) map<string,double>;
  %template(mapii) map<int,int>;
  %template(seti) set<int>;
}

%{
#include "GridForceTypes.h"
#include "GridData.h"
#include "DesolvationGrid.h"
#include "GridForce.h"
#include "GridForceKernels.h"
#include "CachedGridData.h"
#include "IsolatedNonbondedForce.h"
#include "IsolatedNonbondedForceKernels.h"
#include "IsolatedBondedForce.h"
#include "IsolatedBondedForceKernels.h"
#include "IsolatedSiteForce.h"
#include "IsolatedSiteForceKernels.h"
#include "GBSAGridForce.h"
#include "GBSAGridForceKernels.h"
#include "IsolatedGBSAForce.h"
#include "IsolatedGBSAForceKernels.h"
#include "BondedHessian.h"
#include "NewtonMinimizer.h"
#include "BATTopology.h"
#ifdef GRIDFORCE_BUILD_CUDA
#include "CudaBATConverter.h"
#include "CudaSmartDartingPool.h"
#endif
#include "MultiGroupHMCIntegrator.h"
#include "MultiGroupHMCKernels.h"
#include "MultiGroupNUTSIntegrator.h"
#include "MultiGroupNUTSKernels.h"
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

// Declare shared_ptr support for GridData, DesolvationGrid (must be outside namespace)
%shared_ptr(GridForcePlugin::GridData)
%shared_ptr(GridForcePlugin::DesolvationGrid)

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

// Interpolation method constants
// Usage: grid.setInterpolationMethod(gfp.INTERP_TRILINEAR)
%constant int INTERP_TRILINEAR          = 0;
%constant int INTERP_TRICUBIC_BSPLINE   = 1;
%constant int INTERP_TRICUBIC_HERMITE   = 2;
%constant int INTERP_TRIQUINTIC_HERMITE = 3;
%constant int INTERP_TRIQUINTIC_BSPLINE = 4;

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
    double groupScalingFactor;
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

/**
 * DesolvationGrid holds HCT grid values and correction terms for GBSA.
 * Used by GBSAGridForce for efficient grid-based solvation calculations.
 */
class DesolvationGrid {
public:
    // Physical constants
    static const double DEFAULT_PROBE_RADIUS;   // 0.14 nm (water probe)
    static const double DIELECTRIC_OFFSET;      // 0.009 nm
    static const int NUM_DERIVATIVES;           // 27 derivatives per point

    DesolvationGrid();
    DesolvationGrid(int nx, int ny, int nz, double spacing,
                    double probeRadius,
                    const std::vector<double>& rThresholds,
                    bool hasDerivatives = false);

    static std::shared_ptr<DesolvationGrid> loadFromFile(const std::string& filename);
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

    double getSpacing() const;

    %apply double& OUTPUT {double& ox};
    %apply double& OUTPUT {double& oy};
    %apply double& OUTPUT {double& oz};
    void getOrigin(double& ox, double& oy, double& oz) const;
    %clear double& ox;
    %clear double& oy;
    %clear double& oz;
    void setOrigin(double x, double y, double z);

    // Grid parameters
    double getProbeRadius() const;
    int getNumBins() const;
    const std::vector<double>& getRThresholds() const;
    int getBinForRadius(double offsetRadius) const;

    // Data accessors (float32 arrays for memory efficiency)
    const std::vector<float>& getHctProbe() const;
    const std::vector<float>& getCorrectionN() const;
    const std::vector<float>& getCorrectionA() const;
    const std::vector<float>& getCorrectionB() const;

    // Memory info
    size_t getMemoryBytes() const;
    int getFloatsPerPoint() const;

    // Derivative support for triquintic/tricubic interpolation
    bool hasDerivatives() const;
    void setHasDerivatives(bool hasDerivs);
    int getNumDerivsPerPoint() const;

    // Data setters for construction
    void setHctProbe(const std::vector<float>& data);
    void setCorrectionN(const std::vector<float>& data);
    void setCorrectionA(const std::vector<float>& data);
    void setCorrectionB(const std::vector<float>& data);
};

/**
 * GBSAGridForce computes Generalized Born solvation using grid-based HCT.
 */
class GBSAGridForce : public OpenMM::Force {
public:
    // Constants
    static const double OBC_ALPHA;
    static const double OBC_BETA;
    static const double OBC_GAMMA;
    static const double DIELECTRIC_OFFSET;
    static const double DEFAULT_SOLUTE_DIELECTRIC;
    static const double DEFAULT_SOLVENT_DIELECTRIC;
    static const double DEFAULT_SA_SURFACE_TENSION;

    GBSAGridForce();

    int getNumAtoms() const;
    void setNumAtoms(int n);

    void setParticles(const std::vector<int>& particles);
    const std::vector<int>& getParticles() const;

    void setAtomParameters(int index, double charge, double radius, double scaleFactor);
    %apply double& OUTPUT {double& charge};
    %apply double& OUTPUT {double& radius};
    %apply double& OUTPUT {double& scaleFactor};
    void getAtomParameters(int index, double& charge, double& radius, double& scaleFactor) const;
    %clear double& charge;
    %clear double& radius;
    %clear double& scaleFactor;

    void setDesolvationGrid(std::shared_ptr<DesolvationGrid> grid);
    std::shared_ptr<DesolvationGrid> getDesolvationGrid() const;
    void loadDesolvationGrid(const std::string& filename);

    void addExclusion(int atom1, int atom2);
    int getNumExclusions() const;
    %apply int& OUTPUT {int& atom1};
    %apply int& OUTPUT {int& atom2};
    void getExclusionParticles(int index, int& atom1, int& atom2) const;
    %clear int& atom1;
    %clear int& atom2;

    int addParticleGroup(const std::string& name, const std::vector<int>& particleIndices);
    int getNumParticleGroups() const;

    double getSoluteDielectric() const;
    void setSoluteDielectric(double dielectric);
    double getSolventDielectric() const;
    void setSolventDielectric(double dielectric);

    bool getIncludeSurfaceArea() const;
    void setIncludeSurfaceArea(bool include);
    double getSurfaceTension() const;
    void setSurfaceTension(double tension);

    int getInterpolationMethod() const;
    void setInterpolationMethod(int method);
    void setBSplinePrefilterOrder(int order);
    int getBSplinePrefilterOrder() const;

    // Energy reporting
    double getGroupEnergy(int groupIndex) const;
    double getGroupLigandDesolvationEnergy(int groupIndex) const;
    std::vector<double> getGroupBornRadii(int groupIndex) const;

    // Hessian
    void computeHessian(OpenMM::Context& context) const;
    std::vector<double> getHessianBlocks(OpenMM::Context& context) const;
    std::vector<double> getFullHessian(OpenMM::Context& context) const;

    %pythoncode %{
    def getHessianMatrix(self, context):
        """
        Compute and return the full 3N x 3N Hessian matrix as a numpy array.

        Uses numerical finite differences of GBSA forces. Captures cross-atom
        coupling through Born radii (moving atom i changes Born radius of atom j).

        Args:
            context: OpenMM Context (must have called getState with forces first)

        Returns:
            numpy.ndarray: Shape (3N, 3N) Hessian matrix. Units: kJ/(mol*nm^2).
        """
        import numpy as np
        self.computeHessian(context)
        flat = np.array(self.getFullHessian(context))
        n = int(np.sqrt(len(flat)))
        return flat.reshape(n, n)

    def getHessianDiagonalBlocks(self, context):
        """
        Compute and return per-atom 3x3 Hessian diagonal blocks.

        Args:
            context: OpenMM Context (must have called getState with forces first)

        Returns:
            numpy.ndarray: Shape (N, 3, 3) array of per-atom Hessian blocks.
        """
        import numpy as np
        self.computeHessian(context)
        flat = np.array(self.getHessianBlocks(context))
        if len(flat) == 0:
            return np.zeros((0, 3, 3))
        n_atoms = len(flat) // 6
        blocks = flat.reshape(n_atoms, 6)
        H = np.zeros((n_atoms, 3, 3))
        H[:, 0, 0] = blocks[:, 0]  # dxx
        H[:, 1, 1] = blocks[:, 1]  # dyy
        H[:, 2, 2] = blocks[:, 2]  # dzz
        H[:, 0, 1] = H[:, 1, 0] = blocks[:, 3]  # dxy
        H[:, 0, 2] = H[:, 2, 0] = blocks[:, 4]  # dxz
        H[:, 1, 2] = H[:, 2, 1] = blocks[:, 5]  # dyz
        return H
    %}

    // Auto grid generation
    void setAutoGenerateGrid(bool enable);
    bool getAutoGenerateGrid() const;

    void setReceptorAtoms(const std::vector<int>& atoms);
    const std::vector<int>& getReceptorAtoms() const;

    void setReceptorPositions(const std::vector<double>& positions);
    const std::vector<double>& getReceptorPositions() const;

    void setReceptorRadii(const std::vector<double>& radii);
    const std::vector<double>& getReceptorRadii() const;

    void setReceptorScaleFactors(const std::vector<double>& scales);
    const std::vector<double>& getReceptorScaleFactors() const;

    void setGridOrigin(double x, double y, double z);
    %apply double& OUTPUT {double& x};
    %apply double& OUTPUT {double& y};
    %apply double& OUTPUT {double& z};
    void getGridOrigin(double& x, double& y, double& z) const;
    %clear double& x;
    %clear double& y;
    %clear double& z;

    void setGridCounts(int nx, int ny, int nz);
    %apply int& OUTPUT {int& nx};
    %apply int& OUTPUT {int& ny};
    %apply int& OUTPUT {int& nz};
    void getGridCounts(int& nx, int& ny, int& nz) const;
    %clear int& nx;
    %clear int& ny;
    %clear int& nz;

    void setGridSpacing(double spacing);
    double getGridSpacing() const;

    void setProbeRadius(double radius);
    double getProbeRadius() const;

    void setRThresholds(const std::vector<double>& thresholds);
    const std::vector<double>& getRThresholds() const;

    void setComputeGridDerivatives(bool compute);
    bool getComputeGridDerivatives() const;

    void setKDEThreshold(double threshold);
    double getKDEThreshold() const;
    void setKDEBandwidth(double bandwidth);
    double getKDEBandwidth() const;
    void setKDEEpsilonB(double epsilon);
    double getKDEEpsilonB() const;

    bool usesPeriodicBoundaryConditions() const;
};

/**
 * IsolatedGBSAForce computes pairwise GBSA solvation for isolated particle groups.
 */
class IsolatedGBSAForce : public OpenMM::Force {
public:
    // GB method for computing Born radii
    enum GBMethod {
        HCT = 0,     // Raw HCT descreening (no OBC correction)
        OBC_II = 1   // OBC-II with tanh correction (production)
    };

    // Mode for receptor contributions
    enum ReceptorMode {
        NONE = 0,     // Ligand-only (no receptor)
        GRID = 1,     // Receptor HCT from desolvation grid
        PAIRWISE = 2  // Full pairwise receptor-ligand HCT
    };

    // Constants
    static const double OBC_ALPHA;
    static const double OBC_BETA;
    static const double OBC_GAMMA;
    static const double DIELECTRIC_OFFSET;
    static const double DEFAULT_SOLUTE_DIELECTRIC;
    static const double DEFAULT_SOLVENT_DIELECTRIC;
    static const double DEFAULT_SA_SURFACE_TENSION;
    static const double NO_CUTOFF;

    IsolatedGBSAForce();

    int getNumAtoms() const;
    void setNumAtoms(int n);

    void setParticles(const std::vector<int>& particles);
    const std::vector<int>& getParticles() const;

    void setAtomParameters(int index, double charge, double radius, double scaleFactor);
    %apply double& OUTPUT {double& charge};
    %apply double& OUTPUT {double& radius};
    %apply double& OUTPUT {double& scaleFactor};
    void getAtomParameters(int index, double& charge, double& radius, double& scaleFactor) const;
    %clear double& charge;
    %clear double& radius;
    %clear double& scaleFactor;

    // GB method
    GBMethod getGBMethod() const;
    void setGBMethod(GBMethod method);

    // Solvent parameters
    double getSoluteDielectric() const;
    void setSoluteDielectric(double dielectric);
    double getSolventDielectric() const;
    void setSolventDielectric(double dielectric);

    // Surface area term
    bool getIncludeSurfaceArea() const;
    void setIncludeSurfaceArea(bool include);
    double getSurfaceTension() const;
    void setSurfaceTension(double tension);

    // Opt-in diagnostic: download Born radii from device per force eval.
    void setDownloadBornRadii(bool enabled);
    bool getDownloadBornRadii() const;

    // Cutoff
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);

    // Receptor locality cutoff
    double getReceptorLocalityCutoff() const;
    void setReceptorLocalityCutoff(double distance);

    // Receptor mode
    ReceptorMode getReceptorMode() const;
    void setReceptorMode(ReceptorMode mode);

    // Grid mode configuration
    void setDesolvationGrid(std::shared_ptr<DesolvationGrid> grid);
    std::shared_ptr<DesolvationGrid> getDesolvationGrid() const;
    void loadDesolvationGrid(const std::string& filename);
    int getInterpolationMethod() const;
    void setInterpolationMethod(int method);

    // Cross-term scalar-field grid (GRID-mode augment)
    bool getComputeCrossTermGrid() const;
    void setComputeCrossTermGrid(bool enable);
    void setCrossTermBinValues(const std::vector<double>& binValues);
    const std::vector<double>& getCrossTermBinValues() const;
    void setReceptorBornRadiiBaseline(const std::vector<double>& radii);
    const std::vector<double>& getReceptorBornRadiiBaseline() const;

    // Pairwise mode configuration
    void setNumReceptorAtoms(int n);
    int getNumReceptorAtoms() const;
    void setReceptorAtomParameters(int index, double charge, double radius, double scaleFactor);
    %apply double& OUTPUT {double& rec_charge};
    %apply double& OUTPUT {double& rec_radius};
    %apply double& OUTPUT {double& rec_scaleFactor};
    void getReceptorAtomParameters(int index, double& rec_charge, double& rec_radius, double& rec_scaleFactor) const;
    %clear double& rec_charge;
    %clear double& rec_radius;
    %clear double& rec_scaleFactor;
    void setReceptorPositions(const std::vector<double>& positions);
    const std::vector<double>& getReceptorPositions() const;

    // Alchemical scaling
    double getGlobalScalingFactor() const;
    void setGlobalScalingFactor(double factor);
    double getGroupScalingFactor(int groupIndex) const;
    void setGroupScalingFactor(int groupIndex, double factor);

    // Particle groups
    int addParticleGroup(const std::string& name, const std::vector<int>& indices);
    int getNumParticleGroups() const;

    // Energy reporting
    double getGroupEnergy(int groupIndex) const;
    std::vector<double> getParticleGroupEnergies() const;
    double getGroupLigandSelfEnergy(int groupIndex) const;
    double getGroupReceptorContribution(int groupIndex) const;
    double getGroupReceptorDesolvation(int groupIndex) const;
    double getGroupCrossTermEnergy(int groupIndex) const;
    std::vector<double> getGroupBornRadii(int groupIndex) const;
    std::vector<double> getGroupAtomEnergies(int groupIndex) const;

    // Surface area energy reporting
    double getGroupLigandSurfaceArea(int groupIndex) const;
    std::vector<double> getGroupAtomSurfaceAreas(int groupIndex) const;
    std::vector<double> getReceptorBornRadii(int groupIndex) const;
    double getGroupReceptorSurfaceAreaChange(int groupIndex) const;

    // Unscaled energies (no per-group alchemical scaling)
    std::vector<double> getParticleGroupUnscaledEnergies(OpenMM::Context& context) const;

    // Hessian
    std::vector<double> computeHessian(OpenMM::Context& context);

    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const;

    %pythoncode %{
    def getHessianMatrix(self, context):
        """
        Compute and return the full Hessian matrix as a numpy array.

        This computes the analytical Hessian (second derivatives) of the
        GBSA potential with respect to all atomic coordinates.

        Args:
            context: OpenMM Context containing current positions

        Returns:
            numpy.ndarray: Shape (3N, 3N) Hessian matrix where N is the number
                           of atoms. Units are kJ/(mol·nm²).

        Example:
            >>> H = gbsa_force.getHessianMatrix(context)
            >>> eigenvalues = np.linalg.eigvalsh(H)
        """
        import numpy as np
        flat = np.array(self.computeHessian(context))
        n = self.getNumAtoms()
        return flat.reshape(3*n, 3*n)
    %}
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

    void setGlobalScalingFactor(double factor);
    double getGlobalScalingFactor() const;

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
    void setRuntimeCap(double cap);
    double getRuntimeCap() const;
    void setEvaluateInVSpace(bool enabled);
    bool getEvaluateInVSpace() const;
    void setOutOfBoundsRestraint(double k);
    double getOutOfBoundsRestraint() const;
    void setEffectiveBounds(double minX, double minY, double minZ,
                            double maxX, double maxY, double maxZ);
    bool hasEffectiveBounds() const;
    void clearEffectiveBounds();
    void setInterpolationMethod(int method);
    int getInterpolationMethod() const;
    void setBSplinePrefilterOrder(int order);
    int getBSplinePrefilterOrder() const;
    void setAdaptiveRegularization(double cReg);
    double getAdaptiveRegularization() const;
    void setRegularizationThreshold(double threshold);
    double getRegularizationThreshold() const;
    void setPrefilterPCGTolerance(double tol);
    double getPrefilterPCGTolerance() const;
    void setPrefilterMaxIterations(int maxIter);
    int getPrefilterMaxIterations() const;
    void setArcsinhScale(double scale);
    double getArcsinhScale() const;
    void setGaussianBlurSigma(double sigma);
    double getGaussianBlurSigma() const;

    void setAutoGenerateGrid(bool enable);
    bool getAutoGenerateGrid() const;
    void setGridType(const std::string& type);
    const std::string& getGridType() const;

    void setGridOrigin(double x, double y, double z);
    void getGridOrigin(double& OUTPUT, double& OUTPUT, double& OUTPUT) const;

    void setComputeDerivatives(bool compute);
    bool getComputeDerivatives() const;
    void setUseDoubleStorage(bool useDouble);
    bool getUseDoubleStorage() const;
    bool hasDerivatives() const;
    const std::vector<double>& getDerivatives() const;
    void setDerivatives(const std::vector<double>& derivs);

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
    void setParticleGroupScalingFactor(int groupIndex, double factor);
    double getParticleGroupScalingFactor(int groupIndex) const;

    void swapParticleGroupPositions(OpenMM::Context& context, int group1, int group2) const;

    void setParticleGroupPositionsFlat(OpenMM::Context& context, int groupIndex,
                                        const std::vector<double>& coords) const;
    std::vector<double> getParticleGroupPositionsFlat(OpenMM::Context& context, int groupIndex) const;

    void setParticleGroupVelocitiesFlat(OpenMM::Context& context, int groupIndex,
                                         const std::vector<double>& vels) const;
    std::vector<double> getParticleGroupVelocitiesFlat(OpenMM::Context& context, int groupIndex) const;

    std::vector<double> getParticleGroupEnergies(OpenMM::Context& context) const;
    std::vector<double> getParticleGroupUnscaledEnergies(OpenMM::Context& context) const;
    std::vector<double> getParticleAtomEnergies(OpenMM::Context& context) const;
    std::vector<int> getParticleOutOfBoundsFlags(OpenMM::Context& context) const;

    // Batch HMC operations
    void drawAndSetGroupVelocities(OpenMM::Context& context,
                                    const std::vector<double>& temperatures,
                                    const std::vector<double>& masses,
                                    unsigned int seed = 0) const;
    std::vector<double> computeGroupKineticEnergies(OpenMM::Context& context,
                                                     const std::vector<double>& masses) const;
    std::vector<int> acceptRejectGroups(OpenMM::Context& context,
                                         const std::vector<double>& positionsBackup,
                                         const std::vector<double>& pe_old,
                                         const std::vector<double>& pe_new,
                                         const std::vector<double>& ke_old,
                                         const std::vector<double>& ke_new,
                                         const std::vector<double>& temperatures,
                                         unsigned int seed = 0) const;
    void setAllParticleGroupScalingFactors(const std::vector<double>& factors);

    void setParticleGroupRuntimeCap(int groupIndex, double cap);
    double getParticleGroupRuntimeCap(int groupIndex) const;
    void setAllParticleGroupRuntimeCaps(const std::vector<double>& caps);
    std::vector<double> getAllParticleGroupRuntimeCaps() const;
    std::vector<float> getParticleGroupAtomRawEnergies(OpenMM::Context& context) const;

    // Hessian (second derivative) computation for normal modes analysis
    void computeHessian(OpenMM::Context& context) const;
    std::vector<double> getHessianBlocks(OpenMM::Context& context) const;

    // Third derivative computation (quintic B-spline method 4 only)
    void computeThirdDerivatives(OpenMM::Context& context) const;
    std::vector<double> getThirdDerivativeBlocks(OpenMM::Context& context) const;

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

    def getThirdDerivativeTensors(self, context):
        """
        Compute and return third derivative blocks as (N, 10) numpy array.

        Only supported for quintic B-spline (method 4) interpolation.

        Components per atom:
            [d3xxx, d3yyy, d3zzz, d3xxy, d3xxz, d3xyy, d3xzz, d3yyz, d3yzz, d3xyz]

        Args:
            context: OpenMM Context that has been evaluated (getState with forces)

        Returns:
            numpy.ndarray: Shape (N, 10) array. Units: kJ/(mol*nm^3).
        """
        import numpy as np
        self.computeThirdDerivatives(context)
        flat = np.array(self.getThirdDerivativeBlocks(context))
        if len(flat) == 0:
            return np.zeros((0, 10))
        n_atoms = len(flat) // 10
        return flat.reshape(n_atoms, 10)

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

    // Alchemical scaling
    double getGlobalScalingFactor() const;
    void setGlobalScalingFactor(double factor);
    double getGroupScalingFactor(int groupIndex) const;
    void setGroupScalingFactor(int groupIndex, double factor);

    // Particle groups
    int addParticleGroup(const std::string& name, const std::vector<int>& indices);
    int getNumParticleGroups() const;

    %apply std::string& OUTPUT {std::string& name};
    %apply std::vector<int>& OUTPUT {std::vector<int>& indices};
    void getParticleGroup(int index, std::string& name, std::vector<int>& indices) const;
    %clear std::string& name;
    %clear std::vector<int>& indices;

    // Per-group energy
    double getGroupEnergy(int groupIndex) const;
    std::vector<double> getParticleGroupEnergies() const;

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
 * IsolatedBondedForce computes bonded interactions (bonds, angles, torsions)
 * for multiple isolated ligands with per-group energy reporting.
 */
class IsolatedBondedForce : public OpenMM::Force {
public:
    IsolatedBondedForce();

    int getNumAtoms() const;
    void setNumAtoms(int numAtoms);

    // Bonds: E = 0.5 * k * (r - r0)^2
    int addBond(int atom1, int atom2, double length, double k);
    int getNumBonds() const;

    %apply int& OUTPUT {int& atom1};
    %apply int& OUTPUT {int& atom2};
    %apply double& OUTPUT {double& length};
    %apply double& OUTPUT {double& k};
    void getBondParameters(int index, int& atom1, int& atom2, double& length, double& k) const;
    %clear int& atom1;
    %clear int& atom2;
    %clear double& length;
    %clear double& k;

    void setBondParameters(int index, int atom1, int atom2, double length, double k);

    // Angles: E = 0.5 * k * (theta - theta0)^2
    int addAngle(int atom1, int atom2, int atom3, double angle, double k);
    int getNumAngles() const;

    %apply int& OUTPUT {int& atom1};
    %apply int& OUTPUT {int& atom2};
    %apply int& OUTPUT {int& atom3};
    %apply double& OUTPUT {double& angle};
    %apply double& OUTPUT {double& k};
    void getAngleParameters(int index, int& atom1, int& atom2, int& atom3,
                            double& angle, double& k) const;
    %clear int& atom1;
    %clear int& atom2;
    %clear int& atom3;
    %clear double& angle;
    %clear double& k;

    void setAngleParameters(int index, int atom1, int atom2, int atom3, double angle, double k);

    // Torsions: E = k * (1 + cos(n*phi - phase))
    int addTorsion(int atom1, int atom2, int atom3, int atom4,
                   int periodicity, double phase, double k);
    int getNumTorsions() const;

    %apply int& OUTPUT {int& atom1};
    %apply int& OUTPUT {int& atom2};
    %apply int& OUTPUT {int& atom3};
    %apply int& OUTPUT {int& atom4};
    %apply int& OUTPUT {int& periodicity};
    %apply double& OUTPUT {double& phase};
    %apply double& OUTPUT {double& k};
    void getTorsionParameters(int index, int& atom1, int& atom2, int& atom3, int& atom4,
                              int& periodicity, double& phase, double& k) const;
    %clear int& atom1;
    %clear int& atom2;
    %clear int& atom3;
    %clear int& atom4;
    %clear int& periodicity;
    %clear double& phase;
    %clear double& k;

    void setTorsionParameters(int index, int atom1, int atom2, int atom3, int atom4,
                              int periodicity, double phase, double k);

    // Alchemical scaling
    double getGlobalScalingFactor() const;
    void setGlobalScalingFactor(double factor);
    double getGroupScalingFactor(int groupIndex) const;
    void setGroupScalingFactor(int groupIndex, double factor);

    // Particle groups
    int addParticleGroup(const std::string& name, const std::vector<int>& indices);
    int getNumParticleGroups() const;

    %apply std::string& OUTPUT {std::string& name};
    %apply std::vector<int>& OUTPUT {std::vector<int>& indices};
    void getParticleGroup(int index, std::string& name, std::vector<int>& indices) const;
    %clear std::string& name;
    %clear std::vector<int>& indices;

    // Per-group energy
    double getGroupEnergy(int groupIndex) const;
    std::vector<double> getParticleGroupEnergies() const;

    void updateParametersInContext(Context &context);

    // Hessian
    std::vector<double> computeHessian(OpenMM::Context& context, int groupIndex = 0);
    std::vector<double> computeInternalForceConstants(OpenMM::Context& context, int groupIndex = 0);
    std::vector<int> getInternalCoordinateAtomIndices() const;

    %pythoncode %{
    def getHessianMatrix(self, context, groupIndex=0):
        """
        Compute and return the bonded Hessian matrix as a numpy array.

        Args:
            context: OpenMM Context containing current positions
            groupIndex: which particle group's positions to use (default 0)

        Returns:
            numpy.ndarray: Shape (3N, 3N) Hessian matrix where N is the
                           template atom count. Units: kJ/(mol*nm^2).
        """
        import numpy as np
        flat = np.array(self.computeHessian(context, groupIndex))
        n = self.getNumAtoms()
        return flat.reshape(3*n, 3*n)

    def getInternalForceConstants(self, context, groupIndex=0):
        """
        Get scalar force constants for each internal DOF.

        Returns:
            dict with keys 'bonds', 'angles', 'torsions', 'all'
        """
        import numpy as np
        fc = np.array(self.computeInternalForceConstants(context, groupIndex))
        nb = self.getNumBonds()
        na = self.getNumAngles()
        return {
            'bonds': fc[:nb],
            'angles': fc[nb:nb+na],
            'torsions': fc[nb+na:],
            'all': fc,
        }

    def getAtomIndicesPerDOF(self):
        """
        Get template atom indices for each internal DOF.

        Returns:
            dict with keys 'bonds', 'angles', 'torsions'
        """
        indices = list(self.getInternalCoordinateAtomIndices())
        nb = self.getNumBonds()
        na = self.getNumAngles()
        nt = self.getNumTorsions()
        offset = 0
        bonds = []
        for b in range(nb):
            bonds.append((indices[offset], indices[offset+1]))
            offset += 2
        angles = []
        for a in range(na):
            angles.append((indices[offset], indices[offset+1], indices[offset+2]))
            offset += 3
        torsions = []
        for t in range(nt):
            torsions.append((indices[offset], indices[offset+1], indices[offset+2], indices[offset+3]))
            offset += 4
        return {'bonds': bonds, 'angles': angles, 'torsions': torsions}
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
    std::vector<double> computeInternalForceConstants(OpenMM::Context& context);
    std::vector<int> getInternalCoordinateAtomIndices() const;
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

    def getInternalForceConstants(self, context):
        """
        Get scalar force constants (d^2E/dq^2) for each internal DOF.

        Returns:
            dict with keys:
                'bonds': numpy array of shape (numBonds,) - always k (kJ/mol/nm^2)
                'angles': numpy array of shape (numAngles,) - always k (kJ/mol/rad^2)
                'torsions': numpy array of shape (numTorsions,) - geometry-dependent (kJ/mol/rad^2)
                'all': concatenated array [bonds, angles, torsions]
        """
        import numpy as np
        fc = np.array(self.computeInternalForceConstants(context))
        nb = self.getNumBonds()
        na = self.getNumAngles()
        return {
            'bonds': fc[:nb],
            'angles': fc[nb:nb+na],
            'torsions': fc[nb+na:],
            'all': fc,
        }

    def getAtomIndicesPerDOF(self):
        """
        Get atom indices for each internal DOF.

        Returns:
            dict with keys:
                'bonds': list of (i, j) tuples
                'angles': list of (i, j, k) tuples
                'torsions': list of (i, j, k, l) tuples
        """
        indices = list(self.getInternalCoordinateAtomIndices())
        nb = self.getNumBonds()
        na = self.getNumAngles()
        nt = self.getNumTorsions()
        offset = 0
        bonds = []
        for b in range(nb):
            bonds.append((indices[offset], indices[offset+1]))
            offset += 2
        angles = []
        for a in range(na):
            angles.append((indices[offset], indices[offset+1], indices[offset+2]))
            offset += 3
        torsions = []
        for t in range(nt):
            torsions.append((indices[offset], indices[offset+1], indices[offset+2], indices[offset+3]))
            offset += 4
        return {'bonds': bonds, 'angles': angles, 'torsions': torsions}

    def getWilsonBMatrix(self, context, system):
        """
        Build the Wilson B-matrix for all internal coordinates at the current geometry.

        Row i of B is dq_i/dx (the gradient of internal coordinate q_i with respect to
        all 3N Cartesian coordinates). Rows are ordered: bonds, angles, torsions, matching
        getAtomIndicesPerDOF(). Bond rows are dimensionless; angle and torsion rows are in
        units of 1/nm. Torsion rows use the Blondel-Karplus formulation (G1, G2, G3, G4).

        Args:
            context: OpenMM Context with current positions
            system: OpenMM System (for n_atoms)

        Returns:
            numpy.ndarray of shape (n_dof, 3*n_atoms)
        """
        import numpy as np

        atom_idx = self.getAtomIndicesPerDOF()
        n_atoms = system.getNumParticles()

        state = context.getState(getPositions=True)
        pos = np.array([[v.x, v.y, v.z] for v in state.getPositions()])

        nb = self.getNumBonds()
        na = self.getNumAngles()
        nt = self.getNumTorsions()
        n_dof = nb + na + nt

        B = np.zeros((n_dof, 3 * n_atoms))
        dof_idx = 0

        # Bond B-matrix rows: dr/dr_i = -rhat, dr/dr_j = rhat
        for b, (i, j) in enumerate(atom_idx['bonds']):
            rij = pos[j] - pos[i]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                dof_idx += 1
                continue
            rhat = rij / r
            B[dof_idx, 3*i:3*i+3] = -rhat
            B[dof_idx, 3*j:3*j+3] = rhat
            dof_idx += 1

        # Angle B-matrix rows
        for a, (i, j, k) in enumerate(atom_idx['angles']):
            r21 = pos[i] - pos[j]
            r23 = pos[k] - pos[j]
            L1 = np.linalg.norm(r21)
            L3 = np.linalg.norm(r23)
            if L1 < 1e-10 or L3 < 1e-10:
                dof_idx += 1
                continue
            e1 = r21 / L1
            e3 = r23 / L3
            cos_theta = np.clip(np.dot(e1, e3), -0.9999999, 0.9999999)
            sin_theta = np.sqrt(1 - cos_theta**2)
            if sin_theta < 1e-10:
                dof_idx += 1
                continue
            g1 = -(e3 - cos_theta * e1) / (L1 * sin_theta)
            g3 = -(e1 - cos_theta * e3) / (L3 * sin_theta)
            g2 = -(g1 + g3)
            B[dof_idx, 3*i:3*i+3] = g1
            B[dof_idx, 3*j:3*j+3] = g2
            B[dof_idx, 3*k:3*k+3] = g3
            dof_idx += 1

        # Torsion B-matrix rows (Blondel-Karplus)
        for t, (i, j, k, l) in enumerate(atom_idx['torsions']):
            b1 = pos[j] - pos[i]
            b2 = pos[k] - pos[j]
            b3 = pos[l] - pos[k]
            m = np.cross(b1, b2)
            nv = np.cross(b2, b3)
            m_sq = np.dot(m, m)
            n_sq = np.dot(nv, nv)
            b2_sq = np.dot(b2, b2)
            if m_sq < 1e-20 or n_sq < 1e-20 or b2_sq < 1e-20:
                dof_idx += 1
                continue
            b2_norm = np.sqrt(b2_sq)
            G1 = m * (b2_norm / m_sq)
            G4 = nv * (-b2_norm / n_sq)
            alpha = np.dot(b1, b2) / b2_sq
            beta = np.dot(b3, b2) / b2_sq
            c1 = -(1.0 + alpha)
            c4 = beta
            d1 = alpha
            d4 = -(1.0 + beta)
            G2 = c1 * G1 + c4 * G4
            G3 = d1 * G1 + d4 * G4
            B[dof_idx, 3*i:3*i+3] = G1
            B[dof_idx, 3*j:3*j+3] = G2
            B[dof_idx, 3*k:3*k+3] = G3
            B[dof_idx, 3*l:3*l+3] = G4
            dof_idx += 1

        return B

    def getEffectiveMasses(self, context, system):
        """
        Effective masses (reciprocal of Wilson G-matrix diagonal) for each internal DOF.

        mu_i = 1 / (B[i,:] @ M_inv @ B[i,:])

        For bonds this is the reduced mass in daltons; for angles and torsions it has units
        of dalton/nm^2 because B is in 1/nm — consistent with the convention used elsewhere
        in this module (omega^2 = k / mu yields rad^2/ps^2 when k is in kJ/(mol*rad^2)).

        Args:
            context: OpenMM Context with current positions
            system: OpenMM System (for masses)

        Returns:
            numpy.ndarray of shape (n_dof,) — effective masses
        """
        import numpy as np
        from openmm import unit as omm_unit

        B = self.getWilsonBMatrix(context, system)
        n_atoms = system.getNumParticles()
        masses = np.array([system.getParticleMass(i).value_in_unit(omm_unit.dalton)
                           for i in range(n_atoms)])
        mass_3n = np.repeat(masses, 3)
        inv_mass = 1.0 / mass_3n
        g_diag = (B ** 2) @ inv_mass
        return np.where(g_diag > 1e-30, 1.0 / np.where(g_diag > 0, g_diag, 1.0), 1e30)

    def computeInternalEntropy(self, context, system, grid_forces=None, temperature=300.0):
        """
        Compute entropy in internal coordinates using scalar force constants
        and the Wilson GF-matrix for effective masses.

        Optionally projects grid Hessian blocks into internal coordinate space
        via the Wilson B-matrix, adding external stiffness contributions.

        Args:
            context: OpenMM Context with current positions
            system: OpenMM System (for masses)
            grid_forces: dict of {name: GridForce} or None. If provided,
                         grid Hessian blocks are projected into internal coords.
            temperature: Temperature in K (default 300)

        Returns:
            dict with:
                'total_classical_entropy_kB': total classical entropy in kB
                'total_quantum_entropy_kB': total quantum entropy in kB
                'bond_entropies_kB': per-bond classical entropy
                'angle_entropies_kB': per-angle classical entropy
                'torsion_entropies_kB': per-torsion classical entropy
                'force_constants': dict from getInternalForceConstants
                'effective_masses': effective mass per DOF in daltons
                'frequencies_cm1': vibrational frequency per DOF in cm^-1
                'n_negative': count of DOF with negative force constants
        """
        import numpy as np

        fc_dict = self.getInternalForceConstants(context)
        n_atoms = system.getNumParticles()
        nb = self.getNumBonds()
        na = self.getNumAngles()
        nt = self.getNumTorsions()
        n_dof = nb + na + nt

        B = self.getWilsonBMatrix(context, system)
        effective_masses = self.getEffectiveMasses(context, system)

        # Start with bonded force constants
        f_total = fc_dict['all'].copy()

        # Project grid Hessian into internal coordinates if provided
        if grid_forces is not None:
            n_at = n_atoms
            for name, gf in grid_forces.items():
                gf.computeHessian(context)
                blocks = np.array(gf.getHessianBlocks(context))
                # Build block-diagonal 3Nx3N grid Hessian, then project:
                # F_grid_internal = B @ H_grid @ B^T
                # Since H_grid is block-diagonal, this simplifies to:
                # F_grid_internal[i,j] = sum_a B[i,3a:3a+3] @ H_a @ B[j,3a:3a+3]
                for a in range(n_at):
                    dxx, dyy, dzz, dxy, dxz, dyz = blocks[6*a:6*a+6]
                    H_a = np.array([[dxx, dxy, dxz],
                                    [dxy, dyy, dyz],
                                    [dxz, dyz, dzz]])
                    for ii in range(n_dof):
                        bi = B[ii, 3*a:3*a+3]
                        if np.dot(bi, bi) < 1e-30:
                            continue
                        # Diagonal contribution only (off-diagonal creates coupling between DOF)
                        f_total[ii] += bi @ H_a @ bi

        # Compute frequencies and entropy
        # omega^2 = f / mu (in internal coords, mass-weighted)
        # For entropy: x = hbar * omega / (kB * T)
        hbar_over_kB = 7.6382  # K*ps
        kB_kJ = 8.314462618e-3  # kJ/(mol*K)

        n_negative = 0
        entropies = np.full(n_dof, np.nan)
        frequencies = np.zeros(n_dof)

        for i in range(n_dof):
            if f_total[i] <= 0:
                n_negative += 1
                continue
            omega_sq = f_total[i] / effective_masses[i]  # rad^2/ps^2
            if omega_sq <= 0:
                n_negative += 1
                continue
            omega = np.sqrt(omega_sq)
            # Frequency in cm^-1: nu_tilde = omega / (2*pi*c)
            # omega in rad/ps, c = 2.998e10 cm/s = 2.998e-2 cm/ps
            frequencies[i] = omega / (2 * np.pi * 2.998e-2)  # cm^-1

            x = hbar_over_kB * omega / temperature
            if x < 1e-10:
                entropies[i] = 1 - np.log(x) if x > 0 else 0
            elif x > 30:
                entropies[i] = x * np.exp(-x)
            else:
                exp_x = np.exp(x)
                entropies[i] = x / (exp_x - 1) - np.log(1 - np.exp(-x))

        valid = np.isfinite(entropies)

        return {
            'total_classical_entropy_kB': float(np.nansum(1 - np.log(hbar_over_kB * np.sqrt(np.maximum(f_total, 1e-30) / effective_masses) / temperature))),
            'total_quantum_entropy_kB': float(np.nansum(entropies)),
            'bond_entropies_kB': entropies[:nb],
            'angle_entropies_kB': entropies[nb:nb+na],
            'torsion_entropies_kB': entropies[nb+na:],
            'force_constants': fc_dict,
            'effective_masses': effective_masses,
            'frequencies_cm1': frequencies,
            'n_negative': n_negative,
            'n_dof': n_dof,
            'n_valid_modes': int(np.sum(valid)),
        }
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

/**
 * BAT (Bond/Angle/Torsion) coordinate topology for smart darting.
 *
 * Mirror of the Python AlGDock/mwe/bat_coords.BATTopology class. Build
 * the topology in Python (where bond-graph parsing is easy), then pass
 * the integer arrays here for the GPU-side kernels to consume.
 *
 * Use `getHash()` to verify that the C++ and Python sides see bit-
 * identical topology data before trusting any kernel output.
 */
class BATTopology {
public:
    BATTopology();

    void setTopology(int nAtoms,
                     const std::vector<int>& root,
                     const std::vector<int>& torsions,
                     const std::vector<int>& perturbableMask,
                     const std::vector<int>& primaryTorsionIdx = {});

    int getNumAtoms() const;
    int getNumTorsions() const;

    const std::vector<int>& getRoot() const;
    const std::vector<int>& getTorsions() const;
    const std::vector<int>& getPerturbableMask() const;
    const std::vector<int>& getPrimaryTorsionIndices() const;

    unsigned long long getHash() const;

    void validate() const;
};

#ifdef GRIDFORCE_BUILD_CUDA
/**
 * CUDA Cartesian <-> BAT coordinate converter. Driven by `BATTopology`.
 * Single-precision FP throughout; deterministic given fixed CUDA driver.
 */
class CudaBATConverter {
public:
    CudaBATConverter();
    void initialize(const BATTopology& topo,
                    OpenMM::Context& context, int maxK = 256);

    int getNumAtoms() const;
    int getNumTorsions() const;

    std::vector<float> cartesianToBATHost(const std::vector<float>& positions,
                                          int K);
    std::vector<float> BATToCartesianHost(const std::vector<float>& bat,
                                          int K);
};

/**
 * Smart-darting target pool with CUDA kernels for nearest-target lookup
 * and BAT-space dart proposal.
 */
class CudaSmartDartingPool {
public:
    CudaSmartDartingPool();
    void initialize(const BATTopology& topo,
                    OpenMM::Context& context,
                    const std::vector<float>& targets_BAT_flat,
                    const std::vector<float>& weights,
                    float epsilon_sq);

    int getNumTargets() const;
    int getNumAtoms() const;
    int getNumHeavy() const;
    float getEpsilonSq() const;

    void setCartesianTargets(const std::vector<int>& heavy_inds,
                              const std::vector<float>& targets_heavy_flat);
    void setCartesianFullTargets(const std::vector<float>& targets_full_flat);

    std::vector<int> findNearestHost(const std::vector<float>& bat, int K);
    std::vector<float> findNearestDistsHost(const std::vector<float>& bat, int K);
    std::vector<float> proposeDartHost(const std::vector<float>& bat,
                                       const std::vector<int>& j_per,
                                       const std::vector<int>& k_per, int K);
    std::vector<int> findNearestCartesianHost(
        const std::vector<float>& pos, int K);
    std::vector<float> findNearestCartesianDistsHost(
        const std::vector<float>& pos, int K);
    std::vector<float> proposeCartesianDartHost(
        const std::vector<float>& pos,
        const std::vector<int>& j_per,
        const std::vector<int>& k_per, int K);
};
#endif

class CalcGridForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {return "CalcGridForce";}
};

class CalcIsolatedNonbondedForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {return "CalcIsolatedNonbondedForce";}
};

class CalcIsolatedGBSAForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {return "CalcIsolatedGBSAForce";}
};

class CalcIsolatedBondedForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {return "CalcIsolatedBondedForce";}
};

class CalcIsolatedSiteForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {return "CalcIsolatedSiteForce";}
};

/**
 * IsolatedSiteForce applies a flat-bottom sphere restraint on per-group
 * center of mass. Keeps each ligand replica within a binding site sphere.
 *
 * E = 0.5 * k * max(0, r_com - maxR)^2
 */
class IsolatedSiteForce : public OpenMM::Force {
public:
    IsolatedSiteForce();

    int getNumAtoms() const;
    void setNumAtoms(int numAtoms);

    // Site parameters
    void setSiteCenter(double x, double y, double z);

    %apply double& OUTPUT {double& x};
    %apply double& OUTPUT {double& y};
    %apply double& OUTPUT {double& z};
    void getSiteCenter(double& x, double& y, double& z) const;
    %clear double& x;
    %clear double& y;
    %clear double& z;

    void setMaxRadius(double maxR);
    double getMaxRadius() const;

    void setForceConstant(double k);
    double getForceConstant() const;

    // Atom masses for COM computation
    void setAtomMasses(const std::vector<double>& masses);
    const std::vector<double>& getAtomMasses() const;

    // Alchemical scaling
    double getGlobalScalingFactor() const;
    void setGlobalScalingFactor(double factor);
    double getGroupScalingFactor(int groupIndex) const;
    void setGroupScalingFactor(int groupIndex, double factor);

    // Particle groups
    int addParticleGroup(const std::string& name, const std::vector<int>& indices);
    int getNumParticleGroups() const;

    %apply std::string& OUTPUT {std::string& name};
    %apply std::vector<int>& OUTPUT {std::vector<int>& indices};
    void getParticleGroup(int index, std::string& name, std::vector<int>& indices) const;
    %clear std::string& name;
    %clear std::vector<int>& indices;

    // Per-group energy
    double getGroupEnergy(int groupIndex) const;
    std::vector<double> getParticleGroupEnergies() const;

    void updateParametersInContext(Context &context);
};

class IntegrateMultiGroupHMCStepKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {return "IntegrateMultiGroupHMCStep";}
};

/**
 * MultiGroupHMCIntegrator: GPU-native multi-group HMC integrator with
 * RESPA multi-timestep integration and per-group temperatures/timesteps.
 *
 * Each call to step(1) performs one complete HMC trial per particle group:
 * draw MB velocities, run RESPA NVE trajectory, Metropolis accept/reject.
 */
class MultiGroupHMCIntegrator : public OpenMM::Integrator {
public:
    enum MomentumRefreshMode { FULL = 0, PARTIAL = 1 };
    enum MetricType { METRIC_IDENTITY = 0, METRIC_SOFTABS = 1, METRIC_BLENDED = 2 };
    enum MetricUpdateMode { METRIC_UPDATE_NONE = 0, METRIC_UPDATE_EVERY_TRAJECTORY = 1 };

    MultiGroupHMCIntegrator(int numGroups, int atomsPerGroup, double stepSize);

    // Group configuration
    int getNumGroups() const;
    int getAtomsPerGroup() const;

    // Per-group temperatures (Kelvin)
    void setGroupTemperature(int group, double temperature);
    double getGroupTemperature(int group) const;
    void setAllGroupTemperatures(const std::vector<double>& temperatures);
    std::vector<double> getAllGroupTemperatures() const;

    // RESPA force group schedule: vector of (forceGroupIndex, substeps) pairs
    void setForceGroupSchedule(const std::vector<std::pair<int,int> >& schedule);
    const std::vector<std::pair<int,int> >& getForceGroupSchedule() const;

    // HMC trajectory length (number of outer RESPA steps per trial)
    void setNumOuterSteps(int steps);
    int getNumOuterSteps() const;

    // Per-group outer step counts (matches AlGDock reference where each
    // state adapts steps_per_trial independently). Groups with shorter
    // steps have dt zeroed for the remaining outer steps.
    void setGroupStepsPerTrial(int group, int steps);
    int getGroupStepsPerTrial(int group) const;
    void setAllGroupStepsPerTrial(const std::vector<int>& steps);
    std::vector<int> getAllGroupStepsPerTrial() const;

    // Per-group timestep (ps)
    void setGroupStepSize(int group, double stepSize);
    double getGroupStepSize(int group) const;
    void setAllGroupStepSizes(const std::vector<double>& stepSizes);
    std::vector<double> getAllGroupStepSizes() const;

    // Momentum refreshment
    void setMomentumRefreshMode(MomentumRefreshMode mode);
    MomentumRefreshMode getMomentumRefreshMode() const;
    void setPartialRefreshAngle(double theta);
    double getPartialRefreshAngle() const;

    // Stability guard
    void setStabilityThreshold(double threshold);
    double getStabilityThreshold() const;

    // Accept/reject results
    bool getGroupAccepted(int group) const;
    std::vector<int> getAllGroupAccepted() const;
    double getGroupAcceptanceRate(int group) const;
    int getGroupAcceptCount(int group) const;
    int getGroupTrialCount(int group) const;
    int getGroupStabilityRejectCount(int group) const;
    std::vector<int> getAllGroupStabilityRejectCounts() const;
    void resetAcceptanceCounts();

    // Diagnostics
    double getGroupDeltaH(int group) const;
    std::vector<double> getAllGroupDeltaH() const;

    // External MC configuration
    void setNumMCTrials(int trials);
    int getNumMCTrials() const;
    void setMCStepSize(double stepSize);
    double getMCStepSize() const;
    void setGroupMCEnabled(int group, bool enabled);
    bool getGroupMCEnabled(int group) const;
    void setAllGroupMCEnabled(const std::vector<int>& enabled);
    std::vector<int> getAllGroupMCEnabled() const;
    int getMCAttempted() const;
    int getMCAccepted() const;
    std::vector<int> getAllGroupMCAccepted() const;
    void resetMCCounts();

    // Riemannian metric configuration
    void setMetricType(MetricType type);
    MetricType getMetricType() const;
    void setMetricUpdateMode(MetricUpdateMode mode);
    MetricUpdateMode getMetricUpdateMode() const;
    void setSoftAbsAlpha(double alpha);
    double getSoftAbsAlpha() const;
    void setMetricBlendFactor(double beta);
    double getMetricBlendFactor() const;
    void setGridHessianWeight(double w);
    double getGridHessianWeight() const;
    std::vector<double> getGroupMetricConditionNumbers() const;

    // External diagonal Hessian injection (e.g., OBC solvation from JAX)
    void setExternalDiagonalHessian(const std::vector<float>& hessian);
    bool hasExternalHessian() const;
    void clearExternalHessian();
    const std::vector<float>& getExternalDiagonalHessian() const;

    // Random number seed
    int getRandomNumberSeed() const;
    void setRandomNumberSeed(int seed);

    // Integrator interface
    void step(int steps);
};

class IntegrateMultiGroupNUTSStepKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {return "IntegrateMultiGroupNUTSStep";}
};

/**
 * MultiGroupNUTSIntegrator: GPU-native multi-group NUTS integrator with
 * RESPA multi-timestep integration and per-group temperatures/timesteps.
 *
 * Implements the No-U-Turn Sampler (Hoffman & Gelman, 2014) adapted for
 * simultaneous multi-group execution. Each call to step(1) performs one
 * complete NUTS trial per particle group with adaptive trajectory length.
 */
class MultiGroupNUTSIntegrator : public OpenMM::Integrator {
public:
    enum MomentumRefreshMode { FULL = 0, PARTIAL = 1 };
    enum MetricType { METRIC_IDENTITY = 0, METRIC_SOFTABS = 1, METRIC_BLENDED = 2 };
    enum MetricUpdateMode { METRIC_UPDATE_NONE = 0, METRIC_UPDATE_EVERY_TRAJECTORY = 1 };

    MultiGroupNUTSIntegrator(int numGroups, int atomsPerGroup, double stepSize);

    // Group configuration
    int getNumGroups() const;
    int getAtomsPerGroup() const;

    // Per-group temperatures (Kelvin)
    void setGroupTemperature(int group, double temperature);
    double getGroupTemperature(int group) const;
    void setAllGroupTemperatures(const std::vector<double>& temperatures);
    std::vector<double> getAllGroupTemperatures() const;

    // RESPA force group schedule: vector of (forceGroupIndex, substeps) pairs
    void setForceGroupSchedule(const std::vector<std::pair<int,int> >& schedule);
    const std::vector<std::pair<int,int> >& getForceGroupSchedule() const;

    // NUTS tree depth
    void setMaxTreeDepth(int depth);
    int getMaxTreeDepth() const;

    // Per-group timestep (ps)
    void setGroupStepSize(int group, double stepSize);
    double getGroupStepSize(int group) const;
    void setAllGroupStepSizes(const std::vector<double>& stepSizes);
    std::vector<double> getAllGroupStepSizes() const;

    // Momentum refreshment
    void setMomentumRefreshMode(MomentumRefreshMode mode);
    MomentumRefreshMode getMomentumRefreshMode() const;
    void setPartialRefreshAngle(double theta);
    double getPartialRefreshAngle() const;

    // Stability guard
    void setStabilityThreshold(double threshold);
    double getStabilityThreshold() const;

    // Tree depth diagnostics
    int getGroupTreeDepth(int group) const;
    std::vector<int> getAllGroupTreeDepths() const;
    double getGroupMeanTreeDepth(int group) const;

    // Divergence diagnostics
    bool getGroupDivergent(int group) const;
    std::vector<int> getAllGroupDivergent() const;

    // Accept/reject results (accept = non-divergent)
    bool getGroupAccepted(int group) const;
    std::vector<int> getAllGroupAccepted() const;
    double getGroupAcceptanceRate(int group) const;
    int getGroupAcceptCount(int group) const;
    int getGroupTrialCount(int group) const;
    int getGroupDivergenceCount(int group) const;
    std::vector<int> getAllGroupDivergenceCounts() const;
    void resetAcceptanceCounts();

    // External MC configuration
    void setNumMCTrials(int trials);
    int getNumMCTrials() const;
    void setMCStepSize(double stepSize);
    double getMCStepSize() const;
    void setGroupMCEnabled(int group, bool enabled);
    bool getGroupMCEnabled(int group) const;
    void setAllGroupMCEnabled(const std::vector<int>& enabled);
    std::vector<int> getAllGroupMCEnabled() const;
    int getMCAttempted() const;
    int getMCAccepted() const;
    std::vector<int> getAllGroupMCAccepted() const;
    void resetMCCounts();

    // GPU tree building toggle
    void setGpuTreeBuilding(bool enabled);
    bool getGpuTreeBuilding() const;

    // Riemannian metric configuration
    void setMetricType(MetricType type);
    MetricType getMetricType() const;
    void setMetricUpdateMode(MetricUpdateMode mode);
    MetricUpdateMode getMetricUpdateMode() const;
    void setSoftAbsAlpha(double alpha);
    double getSoftAbsAlpha() const;
    void setMetricBlendFactor(double beta);
    double getMetricBlendFactor() const;
    void setGridHessianWeight(double w);
    double getGridHessianWeight() const;
    std::vector<double> getGroupMetricConditionNumbers() const;

    // External diagonal Hessian injection (e.g., OBC solvation from JAX)
    void setExternalDiagonalHessian(const std::vector<float>& hessian);
    bool hasExternalHessian() const;
    void clearExternalHessian();
    const std::vector<float>& getExternalDiagonalHessian() const;

    // Random number seed
    int getRandomNumberSeed() const;
    void setRandomNumberSeed(int seed);

    // Integrator interface
    void step(int steps);
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

def setGridCacheMaxHostMemory(bytes):
    """
    Set the maximum host memory for cached grid data.

    When the cache exceeds this limit, least-recently-used grid entries
    are evicted. Set to 0 for unlimited (default).

    Args:
        bytes: Maximum memory in bytes. E.g., 16 * 1024**3 for 16 GB.
    """
    _gridforceplugin.setGridCacheMaxHostMemory(bytes)

def getGridCacheHostMemoryUsage():
    """
    Get the current host memory usage of the grid data cache in bytes.
    """
    return _gridforceplugin.getGridCacheHostMemoryUsage()

def getGridCacheMaxHostMemory():
    """
    Get the configured maximum host memory for the grid cache in bytes.
    Returns 0 if unlimited.
    """
    return _gridforceplugin.getGridCacheMaxHostMemory()
%}

// Expose cache clearing and memory management functions
%inline %{
void clearGridCache() {
    GridForcePlugin::GridDataCache::clearAll();
}

void setGridCacheMaxHostMemory(size_t bytes) {
    GridForcePlugin::GridDataCache::setMaxHostMemory(bytes);
}

size_t getGridCacheHostMemoryUsage() {
    return GridForcePlugin::GridDataCache::getHostMemoryUsage();
}

size_t getGridCacheMaxHostMemory() {
    return GridForcePlugin::GridDataCache::getMaxHostMemory();
}
%}

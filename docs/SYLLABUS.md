# Introduction to Lattice QCD - Course Syllabus

## Course Information

**Course Title**: Introduction to Lattice Quantum Chromodynamics  
**Duration**: 8 weeks (56 hours total)  
**Prerequisites**: Quantum mechanics, statistical mechanics, basic programming (Python)  
**Level**: Advanced undergraduate / Beginning graduate  

## Course Description

This course provides a comprehensive introduction to lattice field theory with emphasis on quantum chromodynamics (QCD). Students will learn both theoretical foundations and computational techniques, progressing from basic path integrals to advanced Monte Carlo methods and modern acceleration techniques including machine learning approaches.

## Learning Outcomes

Upon successful completion of this course, students will be able to:

1. **Understand** the path integral formulation of quantum mechanics and field theory
2. **Implement** Monte Carlo algorithms for quantum systems from scratch
3. **Analyze** statistical data with proper error estimation techniques
4. **Recognize** and mitigate critical slowing down in simulations
5. **Apply** lattice methods to specific field theories
6. **Evaluate** modern acceleration techniques including machine learning
7. **Design** computational studies of quantum field theories
8. **Communicate** results through clear scientific writing and presentations

## Course Structure

### Phase 1: Foundations (Weeks 1-2)

#### Week 1: Path Integral Formalism
**Lectures** (6 hours):
- L1.1: From Schrödinger to Feynman - Path integral basics
- L1.2: Euclidean path integrals and statistical mechanics connection
- L1.3: Discretization and lattice formulation

**Laboratory** (4 hours):
- Lab 1.1: Path integral for free particle (Exercise 1.1)
- Lab 1.2: Harmonic oscillator implementation

**Assignments**:
- Problem Set 1: Theoretical exercises on path integrals
- Programming Assignment 1: Free particle path integral

**Readings**:
- Creutz, Chapters 1-2
- Course notes: Theory.md sections 1-2

#### Week 2: Quantum Mechanics on the Lattice
**Lectures** (6 hours):
- L2.1: Harmonic oscillator - analytical vs numerical
- L2.2: Action formulation and boundary conditions
- L2.3: Observable measurement and quantum averages

**Laboratory** (4 hours):
- Lab 2.1: Harmonic oscillator optimization
- Lab 2.2: Comparison with exact results

**Assignments**:
- Problem Set 2: Lattice quantum mechanics
- Programming Assignment 2: Harmonic oscillator analysis

**Readings**:
- Creutz, Chapter 3
- Course notes: Harmonic oscillator section

### Phase 2: Monte Carlo Methods (Weeks 3-4)

#### Week 3: Markov Chain Monte Carlo
**Lectures** (6 hours):
- L3.1: Monte Carlo basics and importance sampling
- L3.2: Markov chains and detailed balance
- L3.3: Metropolis-Hastings algorithm

**Laboratory** (4 hours):
- Lab 3.1: Metropolis for Gaussian distribution
- Lab 3.2: Parameter optimization and acceptance rates

**Assignments**:
- Problem Set 3: MCMC theory
- Programming Assignment 3: Metropolis implementation

**Readings**:
- Creutz, Chapters 4-5
- Newman & Barkema, Chapters 1-3

#### Week 4: Statistical Analysis
**Lectures** (6 hours):
- L4.1: Autocorrelation functions and integrated times
- L4.2: Error analysis - jackknife and bootstrap
- L4.3: Effective sample sizes and convergence

**Laboratory** (4 hours):
- Lab 4.1: Statistical analysis tools
- Lab 4.2: Error estimation methods

**Assignments**:
- Problem Set 4: Statistical analysis
- Programming Assignment 4: Analysis utilities

**Readings**:
- Berg, Chapters 1-2
- Course notes: Utils implementation

### Phase 3: Advanced Techniques (Weeks 5-6)

#### Week 5: Critical Slowing Down
**Lectures** (6 hours):
- L5.1: Phase transitions and critical phenomena
- L5.2: Critical slowing down and dynamic exponents
- L5.3: Traditional acceleration methods

**Laboratory** (4 hours):
- Lab 5.1: 2D Ising model implementation (Exercise 2.2)
- Lab 5.2: Critical slowing down measurement

**Assignments**:
- Problem Set 5: Critical phenomena
- Programming Assignment 5: Ising model study

**Readings**:
- Course notes: Critical_Slowing_Down.md sections 1-3
- Landau & Binder, Chapter 4

#### Week 6: Modern Acceleration Methods
**Lectures** (6 hours):
- L6.1: Hybrid Monte Carlo algorithm
- L6.2: Machine learning approaches - flows and GANs
- L6.3: Future directions and advanced techniques

**Laboratory** (4 hours):
- Lab 6.1: HMC implementation and optimization
- Lab 6.2: Introduction to neural network acceleration

**Assignments**:
- Problem Set 6: Advanced algorithms
- Programming Assignment 6: HMC vs Metropolis comparison

**Readings**:
- Course notes: HMC implementation
- Duane et al., "Hybrid Monte Carlo" (1987)

### Phase 4: Applications (Weeks 7-8)

#### Week 7: Field Theory Applications
**Lectures** (6 hours):
- L7.1: Scalar field theory in 1D
- L7.2: Phase transitions and finite size scaling
- L7.3: Correlation functions and physical observables

**Laboratory** (4 hours):
- Lab 7.1: φ⁴ theory implementation
- Lab 7.2: Phase diagram exploration

**Assignments**:
- Problem Set 7: Field theory
- Programming Assignment 7: Critical behavior study

**Readings**:
- Gattringer & Lang, Chapters 1-2
- Course notes: Field theory section

#### Week 8: Project Presentations and Advanced Topics
**Lectures** (4 hours):
- L8.1: Gauge theories and QCD introduction
- L8.2: Current research frontiers

**Laboratory** (2 hours):
- Final project presentations

**Final Projects** (4 hours):
- Student presentations of final projects

## Assessment Structure

### Continuous Assessment (60%)

#### Weekly Reports (40%)
Each week includes a written report covering:
- **Theory Summary** (25%): Understanding of physics concepts
- **Implementation Description** (35%): Code explanation and methodology  
- **Results Analysis** (40%): Data interpretation and conclusions

#### Programming Assignments (20%)
- Working code with proper documentation
- Efficiency and clarity of implementation
- Correct physics implementation

### Final Project (40%)

#### Project Options:
1. **Original Research**: Implement new algorithm or study novel system
2. **Advanced Analysis**: Deep investigation of critical phenomena
3. **Machine Learning**: Develop ML-accelerated sampling methods
4. **Comparison Study**: Systematic algorithm comparison

#### Project Components:
- **Proposal** (Week 4): 2-page research proposal
- **Progress Report** (Week 6): Interim results and methodology
- **Final Report** (Week 8): 10-15 page research paper
- **Presentation** (Week 8): 15-minute conference-style talk

### Grading Rubric

| Component | Weight | Excellent (90-100%) | Good (80-89%) | Satisfactory (70-79%) | Needs Work (<70%) |
|-----------|--------|-------------------|---------------|---------------------|------------------|
| **Physics Understanding** | 30% | Deep insight, theory-computation connection | Good grasp of concepts | Basic understanding | Missing key concepts |
| **Programming Skills** | 25% | Clean, efficient, well-documented | Mostly clean, functional | Works but improvable | Buggy or incomplete |
| **Data Analysis** | 25% | Thorough statistical analysis | Good analysis, minor gaps | Basic analysis complete | Insufficient analysis |
| **Communication** | 20% | Clear, professional presentation | Good communication | Adequate presentation | Poor communication |

## Required Software and Resources

### Software Requirements
- **Python 3.7+** with NumPy, SciPy, Matplotlib
- **Jupyter Notebook** for interactive exercises
- **Git** for version control
- **Optional**: TensorFlow/PyTorch for ML exercises

### Computing Resources
- Personal laptop sufficient for most exercises
- Cluster access available for large-scale simulations
- GPU access for machine learning projects

### Textbooks and References

#### Required Texts:
1. **Creutz, M.** "Quarks, Gluons and Lattices" (Cambridge, 1983)
2. **Course Notes** (provided): Theory, exercises, and implementation guides

#### Recommended References:
1. **Gattringer, C. & Lang, C.B.** "Quantum Chromodynamics on the Lattice" (Springer, 2010)
2. **Berg, B.A.** "Markov Chain Monte Carlo Simulations" (World Scientific, 2004)
3. **Newman, M.E.J. & Barkema, G.T.** "Monte Carlo Methods in Statistical Physics" (Oxford, 1999)

#### Research Papers (provided as needed):
- Duane et al., "Hybrid Monte Carlo", Phys. Lett. B 195, 216 (1987)
- Swendsen & Wang, "Nonuniversal critical dynamics", Phys. Rev. Lett. 58, 86 (1987)
- Modern ML papers for advanced students

## Support and Office Hours

### Instructor Support
- **Office Hours**: Tuesdays & Thursdays, 2:00-4:00 PM
- **Email**: Available for urgent questions
- **Online Forum**: 24/7 discussion board for peer interaction

### Teaching Assistant
- **Lab Sessions**: All laboratory sessions supervised
- **Help Sessions**: Fridays 1:00-3:00 PM for programming help
- **Code Reviews**: Available by appointment

### Technical Support
- **IT Helpdesk**: For computational issues
- **Cluster Support**: For high-performance computing questions

## Course Policies

### Attendance
- **Lectures**: Strongly recommended (slides provided)
- **Laboratory Sessions**: Mandatory (make-up sessions available)
- **Final Presentations**: Mandatory

### Late Work Policy
- **Programming Assignments**: 10% penalty per day late
- **Reports**: 15% penalty per day late  
- **Final Project**: No extensions except documented emergencies

### Academic Integrity
- **Collaboration**: Encouraged on concepts, forbidden on code/reports
- **Citation**: All sources must be properly cited
- **Code Sharing**: Discuss algorithms, don't share implementations

### Accommodation Policy
Students with documented disabilities should contact the disability services office for accommodations.

## Course Schedule (8 Weeks)

| Week | Dates | Topic | Major Deliverables |
|------|--------|-------|-------------------|
| 1 | Jan 8-12 | Path Integral Formalism | Problem Set 1, Programming Assignment 1 |
| 2 | Jan 15-19 | Quantum Mechanics on Lattice | Problem Set 2, Programming Assignment 2 |
| 3 | Jan 22-26 | Markov Chain Monte Carlo | Problem Set 3, Programming Assignment 3 |
| 4 | Jan 29-Feb 2 | Statistical Analysis | Problem Set 4, Programming Assignment 4, **Project Proposal** |
| 5 | Feb 5-9 | Critical Slowing Down | Problem Set 5, Programming Assignment 5 |
| 6 | Feb 12-16 | Modern Acceleration | Problem Set 6, Programming Assignment 6, **Progress Report** |
| 7 | Feb 19-23 | Field Theory Applications | Problem Set 7, Programming Assignment 7 |
| 8 | Feb 26-Mar 2 | Projects & Advanced Topics | **Final Report**, **Presentations** |

## Additional Opportunities

### Research Projects
Outstanding students may be invited to continue research projects beyond the course duration.

### Conference Presentation
Best final projects may be submitted to student conferences or workshops.

### Graduate School Preparation
Course provides excellent preparation for PhD studies in theoretical physics or computational science.

## Course Evaluation

### Student Feedback
- **Mid-term Evaluation** (Week 4): Course adjustments based on feedback
- **Final Evaluation**: Comprehensive course assessment

### Continuous Improvement
Course content updated annually based on:
- Student feedback and performance
- Advances in computational techniques
- New research developments

---

*This syllabus is subject to modification as needed. Students will be notified of any changes in advance.*

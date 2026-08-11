# Background

This task is a chip macro-placement optimization problem.

In a VLSI chip design, a circuit is represented as a netlist: modules are connected by nets, and each movable macro has a physical size and must be placed on a fixed two-dimensional chip canvas. The placement of macros strongly affects the later placement of standard cells, routing congestion, timing, power, and area.

The optimizer controls a vector x that represents proposed macro coordinates on the chip canvas. In the MGO formulation used by this task, x is not directly treated as the final physical layout. Instead, the evaluator decodes the proposed coordinates into a legal macro placement by moving macros to valid grid locations while trying to minimize incremental wirelength.

The returned objective y is the half-perimeter wirelength, HPWL, of the decoded macro placement. Lower y means a better placement. HPWL is a proxy for downstream chip quality: it approximates routing wirelength, but it is cheaper to evaluate than full global placement or final PPA metrics.

This is a black-box optimization task because the optimizer does not have access to a closed-form, differentiable mapping from proposed coordinates x to placement quality y. The final score is produced by a placement evaluator that applies domain-specific decoding, legality handling, and wirelength computation.

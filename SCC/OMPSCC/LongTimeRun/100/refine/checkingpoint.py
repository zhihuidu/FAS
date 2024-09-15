import gurobipy as gp
from gurobipy import GRB

def save_checkpoint(model, filename):
    """Save the current MIP start file."""
    # Save the current solution to a .mst file
    model.write(filename)

def solve_with_checkpoint(graph, checkpoint_file=None):
    model = gp.Model("CheckpointExample")

    # Add variables, constraints, and objective as usual
    # For this example, we add simple variables and constraints
    x = model.addVar(vtype=GRB.CONTINUOUS, name="x")
    y = model.addVar(vtype=GRB.CONTINUOUS, name="y")
    model.setObjective(x + y, GRB.MAXIMIZE)
    model.addConstr(x + 2 * y <= 4, "c0")
    model.addConstr(2 * x + y <= 5, "c1")

    # If there's a checkpoint file, load it
    if checkpoint_file:
        print(f"Loading checkpoint from {checkpoint_file}")
        model.read(checkpoint_file)

    # Optimize the model
    model.optimize()

    # Save checkpoint if optimization is interrupted
    if model.status == GRB.INTERRUPTED or model.status == GRB.TIME_LIMIT:
        save_checkpoint(model, 'checkpoint.mst')

    return model

# Usage example
model = solve_with_checkpoint(None)

# Save the checkpoint
save_checkpoint(model, "checkpoint.mst")

model.read('checkpoint.mst')
model.optimize()


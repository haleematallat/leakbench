TARGET = "diagnosis"
FEATURES = [f"x{i}" for i in range(8)]
GROUP_KEY = "visit_id"  # keep records from the same encounter together
TEST_SIZE = 0.3

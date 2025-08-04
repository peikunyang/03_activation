import glob

results = []

for filepath in glob.glob("../*/test?/accu_sum"):
    with open(filepath, "r") as f:
        lines = f.readlines()

    if len(lines) <= 1:
        continue

    data = []
    for line in lines[1:]:
        if line.strip() == "":
            continue
        tokens = line.split()
        if len(tokens) < 3:
            continue
        epoch = int(tokens[0])
        train = float(tokens[1])
        test = float(tokens[2])
        data.append((epoch, train, test))

    if not data:
        continue

    parts = filepath.split("/")
    folder = parts[1]
    testdir = parts[2]
    best = max(data, key=lambda x: x[1])
    results.append((folder, testdir, best[0], best[1], best[2]))

results.sort(key=lambda x: x[3], reverse=True)

with open("rmsd", "w") as output_file:
    for r in results:
        output_file.write(f"{r[0]:<10} {r[1]:<7} {r[2]:>3}  {r[3]:6.2f}  {r[4]:6.2f}\n")


from math import exp, log, sqrt

K = 374750
eps = 0.5
delta = 0.01
expdelta = exp(delta)
data = []

with open("batch-log.txt") as f:
    line = f.readline().strip()
    while True:
        if line == "end":
            break
        if line.startswith("Node"):
            l = line.find('(')
            r = line.find(')')
            pred = float(line[l:r].split(' ', 1)[1])
            g = (exp(pred) - 1) / (exp(pred) + 1)

            line = f.readline().strip()
            line = f.readline().strip()
            track = []
            while line:
                j, cnt, score, w1, wsq, w = line.split(", ")
                j, cnt, score, w1, wsq, w = \
                    int(j), int(cnt), float(score), float(w1), float(wsq), float(w)
                track.append((j, cnt, score, w1, wsq, w))
                line = f.readline().strip()

            j, cnt, score, w1, wsq, w = track[-1]
            oracle = abs(score) / w
            found = False
            i = 0
            while i < len(track) and not found:
                j, cnt, score, w1, wsq, w = track[i]
                if wsq > expdelta:
                    empr = (score - sqrt(wsq * log(log(wsq) / delta))) / w
                else:
                    empr = 0.0
                if empr >= oracle * eps: # and wsq > 0 and w1 * w1 / wsq > 1000:
                    data.append((j, empr, oracle, (j, cnt, score, w1, wsq, w), track[-1]))
                    found = True
                i = i + 1
            if not found:
                data.append((K, empr, oracle, None, track[-1]))
        line = f.readline().strip()

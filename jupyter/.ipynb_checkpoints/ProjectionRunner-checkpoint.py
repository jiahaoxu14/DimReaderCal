import sys
import DualNum
import csv
import numpy as np
import tSNE
import multiprocessing
import datetime
import time
import Grid
import json

class ProjectionRunner:
    def __init__(self,projection,params=None):
        self.params = params
        self.projection = projection
        self.firstRun = False

    def calculateValues(self, points, perturbations=None):


        self.points = points
        self.origPoints = points
        self.resultVect = [0] * (len(self.points) * 2)

        n = len(points)
        self.dualNumPts = DualNum.DualNum(np.array(points), np.zeros(np.shape(points)))

        self.perturb = perturbations
        if not self.firstRun:
            #initial run of tsne to get seed parameters
            p = self.projection(points, self.params)
            p.run()
            self.firstRun = True
            self.runParams= p.getExecParams()


        if (n != 0):
            procs = []
            cpus = multiprocessing.cpu_count()
            xOutArray = multiprocessing.Array('d', range(n))
            yOutArray = multiprocessing.Array('d', range(n))
            outDotArray = multiprocessing.Array('d', range(2 * n))

            if (cpus > n):
                cpus = 1
            chunksize = int(np.floor(float(n) / cpus))
            for i in range(cpus):
                minI = chunksize * i

                if (i < cpus - 1):
                    maxI = chunksize * (i + 1)
                else:
                    maxI = n

                procs.append(multiprocessing.Process(target=self.loopFunc,
                                                     args=(self.dualNumPts, minI, maxI, outDotArray,
                                                            self.runParams, xOutArray, yOutArray)))

            for proc in procs:
                proc.start()

            for proc in procs:
                proc.join()


            points = []
            self.resultVect = [0] * (2 * n)
            for i in range(n):
                self.resultVect[i] = outDotArray[i]
                self.resultVect[i + n] = outDotArray[i + n]
                points.append([xOutArray[i], yOutArray[i]])

        self.points = points

    def loopFunc(self, pts, minI, maxI, dotArr, runParams, xPts, yPts):
        for i in range(minI, maxI):

            if self.perturb is None:
                pts.dot[i][self.axis] = 1
            else:
                pts.dot[i] = self.perturb[i]
            m = len(self.origPoints[0])
            p = self.projection(self.dualNumPts,self.params)
            p.setExecParams(runParams)


            results = p.run()

            print("Min: ", minI, "I: ", i, "max: ", maxI)

            dotArr[2 * i] = results[i][0].dot
            dotArr[2 * i + 1] = results[i][1].dot
            n = len(pts.val)

            if i == 0:
                for j in range(len(self.points)):
                    xPts[j] = results[j][0].val
                    yPts[j] = results[j][1].val
            pts.dot[i] = np.zeros(m)

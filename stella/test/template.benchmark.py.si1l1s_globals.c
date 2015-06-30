/* 
 * Copyright 2013-2015 David Mohr
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/
#include <stdio.h>
#include <stdlib.h>
#define lint // get rid of unused variable warning in mtwist.c
#include "../mtpy/mtwist-1.1/mtwist.c"
#include <math.h>

int K = 10;
double {{rununtiltime_init}};
int {{seed_init}};
double koffp = 1.0;
double kcat = 0.1;

double t = 0.0;
double delta;
int leg = 0;
int substrate = 0;
int obs_i = 0;
int * observations;
const double EXPSTART = 0.2;
double next_obs_time;

double uniform() {
    return mt_drand();
}

double mtpy_exp(double p) {
    double u = 1.0 - uniform();
    return -log(u)/p;
}

double getNextObsTime() {
    //"""Called from run()"""
    //global obs_i, EXPSTART, rununtiltime, delta

    if (obs_i == 0) {
        return EXPSTART;
    }
    if (obs_i==K-1) {
        return rununtiltime;
    }

    return exp(log(EXPSTART)+delta*obs_i);
}

void makeObservation() {
    observations[obs_i] = leg;
    obs_i += 1;

    next_obs_time = getNextObsTime();
}

void step() {
    //"""Called from run()"""
    //global leg, substrate
    if (leg == 0)
        leg += 1;
    else {
        double u1 = uniform();
        if (u1 < 0.5)
            leg -= 1;
        else
            leg += 1;
    }
    if (leg == substrate)
        substrate += 1;
}

int isNextObservation() {
    //global t, next_obs_time, obs_i, K
    return t > next_obs_time && obs_i < K;
}

void run() {
    //global t, next_obs_time, obs_i, K, rununtiltime, leg, substrate
    next_obs_time = getNextObsTime();

    // TODO: Declaring R here is not necessary in Python! But llvm needs a
    // it because otherwise the definition of R does not dominate the use below.
    double R = 0.0;
    while (obs_i < K && t < rununtiltime) {
        if (leg < substrate)
            R = koffp;
        else
            R = kcat;
        t += mtpy_exp(R);

        while (isNextObservation()) {
            makeObservation();
        }

        step();
    }
}

int main(int argc, char ** argv) {
    mt_seed32new(seed);
    delta = (log(rununtiltime)-log(EXPSTART))/(double)(K-1);
    observations = (int *) malloc (sizeof(int) * K);

    run();

    int i;
    printf ("[");
    for (i=0; i<K-1; i++) {
        printf ("%d ", observations[i]);
    }
    printf ("%d]\n", observations[i]);

    free(observations);

    exit (0);
}


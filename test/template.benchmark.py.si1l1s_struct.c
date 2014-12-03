#include <stdio.h>
#include <stdlib.h>
#define lint // get rid of unused variable warning in mtwist.c
#include "../mtpy/mtwist-1.1/mtwist.c"
#include <math.h>

typedef struct {
    int K;
    double {{rununtiltime_decl}};
    int {{seed_decl}};
    double koffp;
    double kcat;

    double t;
    double delta;
    int leg;
    int substrate;
    int obs_i;
    int * observations;
    double next_obs_time;
} spider_t;
const double EXPSTART = 0.2;

double uniform() {
    return mt_drand();
}

double mtpy_exp(double p) {
    double u = 1.0 - uniform();
    return -log(u)/p;
}

double getNextObsTime(spider_t *sp) {
    //"""Called from run()"""
    //global obs_i, EXPSTART, rununtiltime, delta

    if (sp->obs_i == 0) {
        return EXPSTART;
    }
    if (sp->obs_i==sp->K-1) {
        return sp->rununtiltime;
    }

    return exp(log(EXPSTART)+sp->delta*sp->obs_i);
}

void makeObservation(spider_t *sp) {
    sp->observations[sp->obs_i] = sp->leg;
    sp->obs_i += 1;

    sp->next_obs_time = getNextObsTime(sp);
}

void step(spider_t *sp) {
    //"""Called from run()"""
    //global leg, substrate
    if (sp->leg == 0)
        sp->leg += 1;
    else {
        double u1 = uniform();
        if (u1 < 0.5)
            sp->leg -= 1;
        else
            sp->leg += 1;
    }
    if (sp->leg == sp->substrate)
        sp->substrate += 1;
}

int isNextObservation(spider_t *sp) {
    //global t, next_obs_time, obs_i, K
    return sp->t > sp->next_obs_time && sp->obs_i < sp->K;
}

void run(spider_t *sp) {
    //global t, next_obs_time, obs_i, K, rununtiltime, leg, substrate
    sp->next_obs_time = getNextObsTime(sp);

    // TODO: Declaring R here is not necessary in Python! But llvm needs a
    // it because otherwise the definition of R does not dominate the use below.
    double R = 0.0;
    while (sp->obs_i < sp->K && sp->t < sp->rununtiltime) {
        if (sp->leg < sp->substrate)
            R = sp->koffp;
        else
            R = sp->kcat;
        sp->t += mtpy_exp(R);

        while (isNextObservation(sp)) {
            makeObservation(sp);
        }

        step(sp);
    }
}

void init(spider_t *sp) {
    sp->K = 10;
    sp->{{rununtiltime_init}};
    sp->{{seed_init}};
    sp->koffp = 1.0;
    sp->kcat = 0.1;
    sp->t = 0.0;
    sp->leg = 0;
    sp->substrate = 0;
    sp->obs_i = 0;
    sp->delta = (log(sp->rununtiltime)-log(EXPSTART))/(double)(sp->K-1);
    sp->observations = (int *) malloc (sizeof(int) * sp->K);

    mt_seed32new(sp->seed);
}

int main(int argc, char ** argv) {
    spider_t sp;

    init(&sp);

    run(&sp);

    int i;
    printf ("[");
    for (i=0; i<sp.K-1; i++) {
        printf ("%d ", sp.observations[i]);
    }
    printf ("%d]\n", sp.observations[i]);

    free(sp.observations);

    exit (0);
}

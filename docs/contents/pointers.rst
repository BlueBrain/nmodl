NMODL "pointers"
================

Mechanisms can refer to values in other mechanisms, e.g. the sodium current
``ina``. Therefore, it supports a notion of "pointer", called `Datum`. A datum
can store a pointer to a double, a stable pointer to a double, integers, or
pointers to anything else.

Integer Variables
-----------------
One important subset of Datum are what could be referred to as genuine pointers
to other doubles. More precisely, pointers to parameters in other mechanisms or
pointers to the parameters associated with each node, e.g. the voltage.

These make up the majority of usecases for Datum; and are considered the
well-mannered subset.

In CoreNEURON this subset of Datums are treated differently. Because CoreNEURON
stores the values these Datums can point to in a single contiguous array of
doubles, the "pointers" can be expressed as indices into this array.

Therefore, this subset of Datums is referred to as "integer variables".




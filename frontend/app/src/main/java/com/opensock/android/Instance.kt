package com.opensock.android

data class Instance(val x: Float, val y: Float, val owner: Int)

data class Pair(val idx1: Int, val idx2: Int)

class Instances(
    val instances: List<Instance>,
    val pairs: List<Pair>,
    val owners: List<String>
) {
    val firstToSecond: Map<Int, Int>

    init {
        firstToSecond = HashMap()

        for (pair in pairs) {
            firstToSecond.put(pair.idx1, pair.idx2)
            firstToSecond.put(pair.idx2, pair.idx1)
        }
    }

    fun getOtherIdx(idx: Int): Int {
        return firstToSecond.getOrDefault(idx, 0)
    }
}
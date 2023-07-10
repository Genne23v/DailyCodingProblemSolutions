const { get } = require('http');
const { sep } = require('path');

/*
 * Q1.
 * Given an array of integers, return a new array such that each element at
 * index i of the new array is the product of all the numbers in the original
 * array except the one at i.
 * For example, if our input was [1, 2, 3, 4, 5], the expected output would be
 * [120, 60, 40, 30, 24]. If our input was [3, 2, 1], the expected output would
 * be [2, 3, 6].
 * Follow-up: what if you can't use division?
 */
function productExceptSelf(nums) {
    const n = nums.length;
    let prefix = new Array(n);
    let suffix = new Array(n);
    let result = new Array(n);

    prefix[0] = 1;
    for (let i = 1; i < n; i++) {
        prefix[i] = prefix[i - 1] * nums[i - 1];
    }

    suffix[n - 1] = 1;
    for (let i = n - 2; i >= 0; i--) {
        suffix[i] = suffix[i + 1] * nums[i + 1];
    }

    for (let i = 0; i < n; i++) {
        result[i] = prefix[i] * suffix[i];
    }

    return result;
}

console.log('========= Q1 =========');
const nums1 = [1, 2, 3, 4, 5];
console.log(productExceptSelf(nums1));

const nums2 = [3, 2, 1];
console.log(productExceptSelf(nums2));
console.log('\n');

/*
 * Q2.
 * Given an array of integers, find the first missing positive integer in linear
 * time and constant space. In other words, find the lowest positive integer
 * that does not exist in the array. The array can contain duplicates and
 * negative numbers as well.
 * For example, the input [3, 4, -1, 1] should give 2. The input [1, 2, 0]
 * should give 3.
 * You can modify the input array in-place.
 */
function findMissingPositive(nums) {
    const n = nums.length;

    // Step 1: Ignore negative numbers and numbers greater than n
    for (let i = 0; i < n; i++) {
        if (nums[i] <= 0 || nums[i] > n) {
            nums[i] = n + 1; // Mark irrelevant numbers
        }
    }

    // Step 2: Rearrange the array using index as a hash
    for (let i = 0; i < n; i++) {
        const num = Math.abs(nums[i]);
        if (num <= n) {
            nums[num - 1] = -Math.abs(nums[num - 1]);
        }
    }

    // Step 3: Find the index of the first positive number (mismatch)
    for (let i = 0; i < n; i++) {
        if (nums[i] > 0) {
            return i + 1;
        }
    }

    return n + 1;
}

console.log('========= Q2 =========');
const missingInt1 = [3, 4, -1, 1];
console.log(findMissingPositive(missingInt1));

const missingInt2 = [1, 2, 0];
console.log(findMissingPositive(missingInt2));
console.log('\n');

/*
 * Q3.
 * An XOR linked list is a more memory efficient doubly linked list. Instead of
 * each node holding next and prev fields, it holds a field named both, which is
 * an XOR of the next node and the previous node. Implement an XOR linked list;
 * it has an add(element) which adds the element to the end, and a get(index)
 * which returns the node at index.
 * If using a language that has no pointers (such as Python), you can assume you
 * have access to get_pointer and dereference_pointer functions that converts
 * between nodes and memory addresses.
 */
class XORNode {
    #_data;
    #_both;

    constructor(data) {
        this.#_data = data;
    }

    get data() {
        return this.#_data;
    }

    get both() {
        return this.#_both;
    }

    set both(both) {
        this.#_both = both;
    }

    hashCode() {
        let hash = 0;
        const str = JSON.stringify(this.#_data);
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = (hash << 5) - hash + char;
            hash |= 0; // Convert to 32bit integer;
        }
        return hash;
    }
}

class XORLinkedList {
    #_head;
    #_tail;
    #_nodeMap;

    constructor() {
        this.#_nodeMap = new Map();
    }

    add(element) {
        const newNode = new XORNode(element);
        if (!this.#_head) {
            this.#_head = newNode;
            this.#_tail = newNode;
        } else {
            newNode.both = this.getPointer(this.#_tail);
            this.#_tail.both = this.#_tail.both ^ this.getPointer(newNode);
            this.#_tail = newNode;
        }
        this.#_nodeMap.set(newNode.hashCode(), newNode);
    }

    get(index) {
        let curr = this.#_head;
        let prev = null;

        for (let i = 0; i < index; i++) {
            const next = this.dereferencePointer(
                curr.both ^ this.getPointer(prev)
            );
            prev = curr;
            curr = next;
        }
        return curr;
    }

    getPointer(node) {
        return node ? node.hashCode() : 0;
    }

    dereferencePointer(pointer) {
        if (pointer === 0) {
            return null;
        }
        return this.#_nodeMap.get(pointer);
    }
}

console.log('========= Q3 =========');
const xorLinkedList = new XORLinkedList();
xorLinkedList.add(10);
xorLinkedList.add(20);
xorLinkedList.add(30);
xorLinkedList.add(40);
xorLinkedList.add(50);

const node1 = xorLinkedList.get(0);
const node2 = xorLinkedList.get(2);
const node3 = xorLinkedList.get(4);

console.log(node1.data);
console.log(node2.data);
console.log(node3.data);
console.log('\n');

/*
 * Q4.
 * Given a list of integers, write a function that returns the largest sum of
 * non-adjacent numbers. Numbers can be 0 or negative.
 * For example, [2, 4, 6, 2, 5] should return 13, since we pick 2, 6, and 5. [5,
 * 1, 1, 5] should return 10, since we pick 5 and 5.
 * Follow-up: Can you do this in O(N) time and constant space?
 */
function largestSumNonAdjacent(nums) {
    let inclusive = 0; // sum including the current element
    let exclusive = 0; // sum excluding the current element

    for (const num of nums) {
        const temp = inclusive;
        inclusive = Math.max(inclusive, exclusive + num);
        exclusive = temp;
    }

    return Math.max(inclusive, exclusive);
}

console.log('========= Q4 =========');
const numsTonFindLargestNonAdjacentSum1 = [2, 4, 6, 2, 5];
const numsTonFindLargestNonAdjacentSum2 = [5, 1, 1, 5];

console.log(largestSumNonAdjacent(numsTonFindLargestNonAdjacentSum1));
console.log(largestSumNonAdjacent(numsTonFindLargestNonAdjacentSum2));
console.log('\n');

/*
 * Q5.
 * There exists a staircase with N steps, and you can climb up either 1 or 2
 * steps at a time. Given N, write a function that returns the number of unique
 * ways you can climb the staircase. The order of the steps matters.
 * For example, if N is 4, then there are 5 unique ways:
 * 1, 1, 1, 1
 * 2, 1, 1
 * 1, 2, 1
 * 1, 1, 2
 * 2, 2
 * What if, instead of being able to climb 1 or 2 steps at a time, you could
 * climb any number from a set of positive integers X? For example, if X = {1,
 * 3, 5}, you could climb 1, 3, or 5 steps at a time.
 */
function countWays(n, X) {
    let dp = new Array(n + 1).fill(0);
    dp[0] = 1;

    for (let i = 1; i <= n; i++) {
        for (let j = 0; j < X.length; j++) {
            if (i >= X[j]) {
                dp[i] += dp[i - X[j]];
            }
        }
    }

    return dp[n];
}

console.log('========= Q5 =========');
const n = 7;
const X = [1, 3, 5];
console.log(
    `Number of unique ways to climb the staircase with ${n} steps using steps from set ${X}: ${countWays(
        n,
        X
    )}`
);
console.log('\n');

/*
 * Q6.
 * Given an integer k and a string s, find the length of the longest substring
 * that contains at most k distinct characters.
 * For example, given s = "abcba" and k = 2, the longest substring with k
 * distinct characters is "bcb".
 */
function longestSubstringWithDistinct(s, k) {
    if (!s || s.length === 0 || k <= 0) {
        return 0;
    }

    let maxLength = 0;
    let start = 0;
    let distinctCount = 0;
    let charCount = new Map();

    for (let end = 0; end < s.length; end++) {
        const c = s.charAt(end);
        charCount.set(c, charCount.get(c) + 1 || 1);

        if (charCount.get(c) == 1) {
            distinctCount++;
        }

        while (distinctCount > k) {
            const startChar = s.charAt(start);
            charCount.set(startChar, charCount.get(startChar) - 1);

            if (charCount.get(startChar) === 0) {
                distinctCount--;
            }
            start++;
        }

        maxLength = Math.max(maxLength, end - start + 1);
    }

    return maxLength;
}

console.log('========= Q6 =========');
const s = 'abcba';
const k = 2;
console.log(
    `Length of the longest substring with ${k} distinct characters: ${longestSubstringWithDistinct(
        s,
        k
    )}`
);
console.log('\n');

/*
 * Q7.
 * Suppose we represent our file system by a string in the following manner:
 * The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:
 * " dir               "
 * "     subdir1       "
 * "     subdir2       "
 * "         file.ext  "
 * The directory dir contains an empty sub-directory subdir1 and a sub-directory
 * subdir2 containing a file file.ext.
 * The string
 * "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"
 * represents:
 * " dir                       "
 * "     subdir1               "
 * "         file1.ext         "
 * "         subsubdir1        "
 * "     subdir2               "
 * "         subsubdir2        "
 * "             file2.ext     "
 * The directory dir contains two sub-directories subdir1 and subdir2. subdir1
 * contains a file file1.ext and an empty second-level sub-directory subsubdir1.
 * subdir2 contains a second-level sub-directory subsubdir2 containing a file
 * file2.ext.
 * We are interested in finding the longest (number of characters) absolute path
 * to a file within our file system. For example, in the second example above,
 * the longest absolute path is "dir/subdir2/subsubdir2/file2.ext", and its
 * length is 32 (not including the double quotes).
 * Given a string representing the file system in the above format, return the
 * length of the longest absolute path to a file in the abstracted file system.
 * If there is no file in the system, return 0.
 * Note:
 * The name of a file contains at least a period and an extension.
 * The name of a directory or sub-directory will not contain a period.
 */
function lengthLongestPath(input) {
    if (!input || input.length === 0) {
        return 0;
    }

    let paths = input.split('\n');
    let stack = [];
    let maxLength = 0;

    for (const path of paths) {
        const level = getLevel(path);
        while (stack.length > level) {
            stack.pop();
        }

        const currLength =
            (stack.length === 0 ? 0 : stack[stack.length - 1]) +
            path.length -
            level +
            1;
        stack.push(currLength);

        if (isFile(path)) {
            maxLength = Math.max(maxLength, currLength - 1);
        }
    }

    return maxLength;
}

function getLevel(path) {
    let level = 0;
    let i = 0;
    while (i < path.length && path.charAt(i) === '\t') {
        level++;
        i++;
    }
    return level;
}

function isFile(path) {
    return path.indexOf('.') !== -1;
}

console.log('========= Q7 =========');
const input1 = 'dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext';
console.log(
    `Length of the longest absolute path: ${lengthLongestPath(input1)}`
);

const input2 =
    'dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext';
console.log(
    `Length of the longest absolute path: ${lengthLongestPath(input2)}`
);
console.log('\n');

/*
 * Q8.
 * Given an array of integers and a number k, where 1 <= k <= length of the
 * array, compute the maximum values of each subarray of length k.
 * For example, given array = [10, 5, 2, 7, 8, 7] and k = 3, we should get: [10,
 * 7, 8, 8], since:
 * 10 = max(10, 5, 2)
 * 7 = max(5, 2, 7)
 * 8 = max(2, 7, 8)
 * 8 = max(7, 8, 7)
 * Do this in O(n) time and O(k) space. You can modify the input array in-place
 * and you do not need to store the results. You can simply print them out as
 * you compute them.
 */
function printMaxOfSubarrays(nums, k) {
    const deque = [];
    const n = nums.length;

    for (let i = 0; i < k; i++) {
        while (deque.length > 0 && nums[i] >= nums[deque[deque.length - 1]]) {
            deque.pop();
        }
        deque.push(i);
    }

    for (let i = k; i < n; i++) {
        console.log(nums[deque[0]]);

        // Remove elements outside the current window
        while (deque.length > 0 && deque[0] <= i - k) {
            deque.shift();
        }

        // Remove smaller elements from the deque
        while (deque.length > 0 && nums[i] >= nums[deque[deque.length - 1]]) {
            deque.pop();
        }

        deque.push(i);
    }

    // Print the maximum of the last subarray
    console.log(nums[deque[0]]);
}

console.log('========= Q8 =========');
const nums = [10, 5, 2, 7, 8, 7];
const kRange = 3;
printMaxOfSubarrays(nums, kRange);
console.log('\n');

/*
 * Q9.
 * Implement regular expression matching with the following special characters:
 * . (period) which matches any single character
 * * (asterisk) which matches zero or more of the preceding element
 * That is, implement a function that takes in a string and a valid regular
 * expression and returns whether or not the string matches the regular
 * expression.
 * For example, given the regular expression "ra." and the string "ray", your
 * function should return true. The same regular expression on the string
 * "raymond" should return false.
 * Given the regular expression ".*at" and the string "chat", your function
 * should return true. The same regular expression on the string "chats" should
 * return false.
 */
function isMatch(text, pattern) {
    if (pattern.length === 0) {
        return text.length === 0;
    }

    const firstMatch =
        text.length > 0 &&
        (pattern.charAt(0) == text.charAt(0) || pattern.charAt(0) == '.');

    if (pattern.length >= 2 && pattern.charAt(1) == '*') {
        return (
            isMatch(text, pattern.substring(2)) ||
            (firstMatch && isMatch(text.substring(1), pattern))
        );
    } else {
        return firstMatch && isMatch(text.substring(1), pattern.substring(1));
    }
}

console.log('========= Q9 =========');
let text = 'ray';
let pattern = 'ra.';
console.log(isMatch(text, pattern)); // true

text = 'raymond';
pattern = 'ra.';
console.log(isMatch(text, pattern)); // false

text = 'chat';
pattern = '.*at';
console.log(isMatch(text, pattern)); // true

text = 'chats';
console.log(isMatch(text, pattern)); // false
console.log('\n');

/*
 * Q10.
 * Suppose you are given a table of currency exchange rates, represented as a 2D
 * array. Determine whether there is a possible arbitrage: that is, whether
 * there is some sequence of trades you can make, starting with some amount A of
 * any currency, so that you can end up with some amount greater than A of that
 * currency.
 * There are no transaction costs and you can trade fractional quantities.
 */
function hasArbitrage(rates) {
    const n = rates.length;
    let dist = new Array(n).fill(Number.MAX_SAFE_INTEGER);
    dist[0] = 0;

    // Relax edges repeatedly (Bellman-Ford algorithm)
    for (let i = 0; i < n - 1; i++) {
        for (let u = 0; u < n; u++) {
            for (let v = 0; v < n; v++) {
                if (dist[u] != Number.MAX_SAFE_INTEGER && rates[u][v] != 0) {
                    const newDist = dist[u] + Math.log(rates[u][v]);
                    if (newDist < dist[v]) {
                        dist[v] = newDist;
                    }
                }
            }
        }
    }

    // Check for negative cycles
    for (let u = 0; u < n; u++) {
        for (let v = 0; v < n; v++) {
            if (dist[u] != Number.MAX_SAFE_INTEGER && rates[u][v] != 0) {
                const newDist = dist[u] + Math.log(rates[u][v]);
                if (newDist < dist[v]) {
                    return true; // Negative cycle exists
                }
            }
        }
    }
    return false;
}

console.log('========= Q10 =========');
const rates = [
    [1.0, 0.05, 0.77],
    [1.18, 1.0, 0.91],
    [1.3, 1.1, 1.0],
];

console.log(`Arbitrage is${hasArbitrage(rates) ? '' : ' not'} possible`);
console.log('\n');

/*
 * Q11.
 * Given an array of strictly the characters 'R', 'G', and 'B', segregate the
 * values of the array so that all the Rs come first, the Gs come second, and
 * the Bs come last. You can only swap elements of the array.
 * Do this in linear time and in-place.
 * For example, given the array ['G', 'B', 'R', 'R', 'B', 'R', 'G'], it should
 * become ['R', 'R', 'R', 'G', 'G', 'B', 'B'].
 */
function segregateColors(colors) {
    let low = 0;
    let mid = 0;
    let high = colors.length - 1;

    while (mid <= high) {
        if (colors[mid] == 'R') {
            swap(colors, low, mid);
            low++;
            mid++;
        } else if (colors[mid] == 'G') {
            mid++;
        } else if (colors[mid] == 'B') {
            swap(colors, mid, high);
            high--;
        }
    }
}

function swap(colors, i, j) {
    const temp = colors[i];
    colors[i] = colors[j];
    colors[j] = temp;
}

console.log('========= Q11 =========');
const colors = ['G', 'B', 'R', 'R', 'B', 'R', 'G'];
segregateColors(colors);
console.log(colors);
console.log('\n');

/*
 * Q12.
 * You have an N by N board. Write a function that, given N, returns the number
 * of possible arrangements of the board where N queens can be placed on the
 * board without threatening each other, i.e. no two queens share the same row,
 * column, or diagonal.
 */
function countQueens(n) {
    let queens = new Array(n).fill(-1);
    return backtrack(queens, 0);
}

function backtrack(queens, row) {
    let count = 0;
    if (row === queens.length) {
        return 1;
    }

    for (let col = 0; col < queens.length; col++) {
        if (isSafe(queens, row, col)) {
            queens[row] = col;
            count += backtrack(queens, row + 1);
        }
    }

    return count;
}

function isSafe(queens, row, col) {
    for (let i = 0; i < row; i++) {
        if (
            queens[i] === col ||
            queens[i] - i === col - row ||
            queens[i] + i === col + row
        ) {
            return false;
        }
    }
    return true;
}

console.log('========= Q12 =========');
const N = 4;
console.log(`Number of possible arrangements: ${countQueens(N)}`);
console.log('\n');

/*
 * Q13.
 * Given an array of integers where every integer occurs three times except for
 * one integer, which only occurs once, find and return the non-duplicated
 * integer.
 * For example, given [6, 1, 3, 3, 3, 6, 6], return 1. Given [13, 19, 13, 13],
 * return 19.
 * Do this in O(N) time and O(1) space.
 */
function findNonDuplicated(nums) {
    let ones = 0; // Count only bits that appear once
    let twos = 0; // Count only bits that appear twice

    for (const num of nums) {
        ones = (ones ^ num) & ~twos;
        twos = (twos ^ num) & ~ones;
    }

    return ones;
}

console.log('========= Q13 =========');
const numsToFindSingleAppearance1 = [6, 1, 3, 3, 3, 6, 6];
const numsToFindSingleAppearance2 = [13, 19, 13, 13];

console.log(
    `Non-duplicated number: ${findNonDuplicated(numsToFindSingleAppearance1)}`
);
console.log(
    `Non-duplicated number: ${findNonDuplicated(numsToFindSingleAppearance2)}`
);
console.log('\n');

/*
 * Q14.
 * Given a list of integers S and a target number k, write a function that
 * returns a subset of S that adds up to k. If such a subset cannot be made,
 * then return null.
 * Integers can appear more than once in the list. You may assume all numbers in
 * the list are positive.
 * For example, given S = [12, 1, 61, 5, 9, 2] and k = 24, return [12, 9, 2, 1]
 * since it sums up to 24.
 */
function findSubset(nums, target) {
    let subset = [];
    const targetFound = backtrack(nums, target, 0, subset);
    return targetFound ? subset : null;
}

function backtrack(nums, target, index, subset) {
    if (target === 0) {
        return true;
    }

    for (let i = index; i < nums.length; i++) {
        if (nums[i] <= target) {
            subset.push(nums[i]);
            const targetFound = backtrack(
                nums,
                target - nums[i],
                i + 1,
                subset
            );

            if (targetFound) {
                return true;
            }
            subset.pop();
        }
    }
    return false;
}

console.log('========= Q14 =========');
const numsToFindSubset = [12, 1, 61, 5, 9, 2];
const target = 24;
console.log(`Subset: ${findSubset(numsToFindSubset, target)}`);
console.log('\n');

/*
 * Q15.
 * Given a string, find the longest palindromic contiguous substring. If there
 * are more than one with the maximum length, return any one.
 * For example, the longest palindromic substring of "aabcdcb" is "bcdcb". The
 * longest palindromic substring of "bananas" is "anana".
 */
function longestPalindromicSubstring(s) {
    if (!s || s.length < 2) {
        return s;
    }

    let start = 0;
    let maxLength = 1;
    const n = s.length;
    let dp = new Array(n).fill(true).map(() => new Array(n).fill(true));

    for (let i = 0; i < n - 1; i++) {
        if (s[i] === s[i + 1]) {
            dp[i][i + 1] = true;
            start = i;
            maxLength = 2;
        }
    }

    for (let len = 3; len <= n; len++) {
        for (let i = 0; i <= n - len; i++) {
            const j = i + len - 1;
            if (s[i] === s[j] && dp[i + 1][j - 1]) {
                dp[i][j] = true;

                if (len > maxLength) {
                    start = i;
                    maxLength = len;
                }
            }
        }
    }
    return s.substring(start, start + maxLength);
}

console.log('========= Q15 =========');
const s1 = 'aabcdcb';
const s2 = 'bananas';

console.log(
    `Longest palindromic substring1: ${longestPalindromicSubstring(s1)}`
);
console.log(
    `Longest palindromic substring2: ${longestPalindromicSubstring(s2)}`
);
console.log('\n');

/*
 * Q16.
 * Implement an LRU (Least Recently Used) cache. It should be able to be
 * initialized with a cache size n, and contain the following methods:
 * set(key, value): sets key to value. If there are already n items in the cache
 * and we are adding a new item, then it should also remove the least recently
 * used item.
 * get(key): gets the value at key. If no such key exists, return null.
 * Each operation should run in O(1) time.
 */
class LRUCacheNode {
    constructor(key, value) {
        this.key = key;
        this.value = value;
        this.prev = null;
        this.next = null;
    }
}

class LRUCache {
    #_capacity;
    #_cache;
    #_head;
    #_tail;

    constructor(capacity) {
        this.#_capacity = capacity;
        this.#_cache = new Map();

        this.#_head = new LRUCacheNode(-1, -1);
        this.#_tail = new LRUCacheNode(-1, -1);
        this.#_head.next = this.#_tail;
        this.#_tail.prev = this.#_head;
    }

    get(key) {
        if (this.#_cache.has(key)) {
            const node = this.#_cache.get(key);
            this.#_removeNode(node);
            this.#_addNodeToHead(node);
            return node.value;
        }
        return -1;
    }

    set(key, value) {
        if (this.#_cache.has(key)) {
            const node = this.#_cache.get(key);
            node.value = value;
            this.#_removeNode(node);
            this.#_addNodeToHead(node);
        } else {
            const newNode = new LRUCacheNode(key, value);
            if (this.#_cache.size >= this.#_capacity) {
                const tailNode = this.#_tail.prev;
                this.#_removeNode(tailNode);
                this.#_cache.delete(this.#_tail.prev.key);
            }
            this.#_cache.set(key, newNode);
            this.#_addNodeToHead(newNode);
        }
    }

    #_removeNode(node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    #_addNodeToHead(node) {
        node.next = this.#_head.next;
        node.prev = this.#_head;
        this.#_head.next.prev = node;
        this.#_head.next = node;
    }
}

console.log('========= Q16 =========');
const cache = new LRUCache(3);
cache.set(1, 10);
cache.set(2, 20);
cache.set(3, 30);

console.log(`Get 1: ${cache.get(1)}`);
console.log(`Get 2: ${cache.get(2)}`);

cache.set(4, 40);

console.log(`Get 1: ${cache.get(1)}`);
console.log(`Get 3: ${cache.get(3)}`);
console.log(`Get 4: ${cache.get(4)}`);
console.log('\n');

/*
 * Q17.
 * Sudoku is a puzzle where you're given a partially-filled 9 by 9 grid with
 * digits. The objective is to fill the grid with the constraint that every row,
 * column, and box (3 by 3 subgrid) must contain all of the digits from 1 to 9.
 * Implement an efficient sudoku solver.
 */
function solveSudoku(grid) {
    for (let row = 0; row < 9; row++) {
        for (let col = 0; col < 9; col++) {
            if (grid[row][col] === 0) {
                for (let digit = 1; digit <= 9; digit++) {
                    if (isValid(grid, row, col, digit)) {
                        grid[row][col] = digit;
                        if (solveSudoku(grid)) {
                            return true;
                        } else {
                            grid[row][col] = 0;
                        }
                    }
                }
                return false;
            }
        }
    }
    return true;
}

function isValid(grid, row, col, digit) {
    for (let i = 0; i < 9; i++) {
        if (grid[row][i] === digit || grid[i][col] === digit) {
            return false;
        }
    }

    const subgridStartRow = Math.floor(row / 3) * 3;
    const subgridStartCol = Math.floor(col / 3) * 3;

    for (let i = subgridStartRow; i < subgridStartRow + 3; i++) {
        for (let j = subgridStartCol; j < subgridStartCol + 3; j++) {
            if (grid[i][j] === digit) {
                return false;
            }
        }
    }
    return true;
}

console.log('========= Q17 =========');
const grid = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
];

if (solveSudoku(grid)) {
    console.log(grid);
} else {
    console.log('No solution exists');
}
console.log('\n');

/*
 * Q18.
 * Implement a file syncing algorithm for two computers over a low-bandwidth
 * network. What if we know the files in the two computers are mostly the same?
 */
console.log('========= Q18 =========');
/*
 * 1. Establish a TCP connection between the two computers.
 * 2. Identify the files that need to be synced. Maintain list of files on each
 * computer and compare the differences.
 * 3. Use a differential syncing algorithm to transfer only different parts.
 * - Implement a mechanism to compare files to identify the differences such as
 * file hashing or timestamp comparison.
 * - When a file is modified, calculate the difference between old and new
 * version. This can be done by comparing contents of the files or using delta
 * encoding techniques.
 * 4. Transfer the differences over the low-bandwidth network. This can be done
 * by sending only the modified parts of the files or using compression
 * techniques to reduce data size.
 * 5. On the receiving computer, apply the received differences to update the
 * corresponding files. This can involve patching or merging the changes into
 * the existing files.
 * 6. Repeat synching process periodically or whenever changes are detected on
 * either computer.
 * 7. Implement error handling and recovery mechanisms to handle network
 * failures, file conflicts, etc.
 * 8. Monitor the syncing process and provide feedback to the users about the
 * progress and status of the synchronization.
 */
console.log('\n');

/*
 * Q19.
 * A knight's tour is a sequence of moves by a knight on a chessboard such that
 * all squares are visited once.
 * Given N, write a function to return the number of knight's tours on an N by N
 * chessboard.
 */
class KnightsTour {
    #_ROWS_MOVES = [2, 1, -1, -2, -2, -1, 1, 2];
    #_COLS_MOVES = [1, 2, 2, 1, -1, -2, -2, -1];

    countKnightTours(n) {
        let board = new Array(n).fill(0).map(() => new Array(n).fill(0));
        let count = 0;

        for (let row = 0; row < n; row++) {
            for (let col = 0; col < n; col++) {
                count += this.#_findTours(board, row, col, 1);
            }
        }
        return count;
    }

    #_findTours(board, row, col, moveCount) {
        const n = board.length;

        if (moveCount === n * n) {
            return 1;
        }

        let count = 0;
        board[row][col] = moveCount;

        for (let i = 0; i < 8; i++) {
            const nextRow = row + this.#_ROWS_MOVES[i];
            const nextCol = col + this.#_COLS_MOVES[i];

            if (this.#_isValidMove(board, nextRow, nextCol)) {
                count += this.#_findTours(
                    board,
                    nextRow,
                    nextCol,
                    moveCount + 1
                );
            }
        }

        board[row][col] = 0;
        return count;
    }

    #_isValidMove(board, row, col) {
        const n = board.length;
        return (
            row >= 0 && row < n && col >= 0 && col < n && board[row][col] === 0
        );
    }
}

console.log('========= Q19 =========');
const knightsTour = new KnightsTour();
console.log(knightsTour.countKnightTours(5));
console.log('\n');

/*
 * Q20.
 * Implement an LFU (Least Frequently Used) cache. It should be able to be
 * initialized with a cache size n, and contain the following methods:
 * set(key, value): sets key to value. If there are already n items in the cache
 * and we are adding a new item, then it should also remove the least frequently
 * used item. If there is a tie, then the least recently used key should be
 * removed.
 * get(key): gets the value at key. If no such key exists, return null.
 * Each operation should run in O(1) time.
 */
class LFUCache {
    #_capacity;
    #_minFrequency;
    #_keyToValue;
    #_keyToFrequency;
    #_frequencyToKeys;

    constructor(capacity) {
        this.#_capacity = capacity;
        this.#_minFrequency = 0;
        this.#_keyToValue = new Map();
        this.#_keyToFrequency = new Map();
        this.#_frequencyToKeys = new Map();
    }

    get(key) {
        if (!this.#_keyToValue.has(key)) {
            return -1;
        }

        const frequency = this.#_keyToFrequency.get(key);
        this.#_keyToFrequency.set(key, frequency + 1);
        this.#_frequencyToKeys.get(frequency).delete(key);

        if (this.#_frequencyToKeys.get(frequency).size === 0) {
            this.#_frequencyToKeys.delete(frequency);

            if (frequency === this.#_minFrequency) {
                this.#_minFrequency++;
            }
        }

        this.#_frequencyToKeys.set(
            frequency + 1,
            this.#_frequencyToKeys.get(frequency + 1)
                ? this.#_frequencyToKeys.get(frequency + 1).add(key)
                : new Set([key])
        );

        return this.#_keyToValue.get(key);
    }

    set(key, value) {
        if (this.#_capacity <= 0) {
            return;
        }

        if (this.#_keyToValue.has(key)) {
            this.#_keyToValue.set(key, value);
            this.get(key);
            return;
        }

        if (this.#_keyToValue.size >= this.#_capacity) {
            const evictKey = this.#_frequencyToKeys
                .get(this.#_minFrequency)
                .values()
                .next().value;
            this.#_frequencyToKeys.get(this.#_minFrequency).delete(evictKey);
            this.#_keyToValue.delete(evictKey);
            this.#_keyToFrequency.delete(evictKey);
        }

        this.#_keyToValue.set(key, value);
        this.#_keyToFrequency.set(key, 1);
        this.#_frequencyToKeys.set(
            1,
            this.#_frequencyToKeys.get(1)
                ? this.#_frequencyToKeys.get(1).add(key)
                : new Set([key])
        );
        this.#_minFrequency = 1;
    }
}

console.log('========= Q20 =========');
const lfuCache = new LFUCache(2);

lfuCache.set(1, 10);
lfuCache.set(2, 20);
console.log(lfuCache.get(1)); // Output: 10

lfuCache.set(3, 30);
console.log(lfuCache.get(2)); // Output: -1
console.log(lfuCache.get(3)); // Output: 30

lfuCache.set(4, 40);
console.log(lfuCache.get(1)); // Output: -1
console.log(lfuCache.get(3)); // Output: 30
console.log(lfuCache.get(4)); // Output: 40
console.log('\n');

/*
 * Q21.
 * In a directed graph, each node is assigned an uppercase letter. We define a
 * path's value as the number of most frequently-occurring letter along that
 * path. For example, if a path in the graph goes through "ABACA", the value of
 * the path is 3, since there are 3 occurrences of 'A' on the path.
 * Given a graph with n nodes and m directed edges, return the largest value
 * path of the graph. If the largest value is infinite, then return null.
 * The graph is represented with a string and an edge list. The i-th character
 * represents the uppercase letter of the i-th node. Each tuple in the edge list
 * (i, j) means there is a directed edge from the i-th node to the j-th node.
 * Self-edges are possible, as well as multi-edges.
 * For example, the following input graph:
 * ABACA
 * [(0, 1),
 * (0, 2),
 * (2, 3),
 * (3, 4)]
 * Would have maximum value 3 using the path of vertices [0, 2, 3, 4], (A, A, C,
 * A).
 * The following input graph:
 * A
 * [(0, 0)]
 * Should return null, since we have an infinite loop.
 */
class DirectedGraph {
    #_adjacencyList;
    #_nodeValues;

    constructor(nodes, edges) {
        this.#_adjacencyList = new Map();
        this.#_nodeValues = new Array(nodes.length);

        for (let i = 0; i < nodes.length; i++) {
            this.#_adjacencyList.set(i, new Set());
            this.#_nodeValues[i] = nodes[i];
        }

        for (const [from, to] of edges) {
            this.#_adjacencyList.get(from).add(to);
        }
    }

    findLargestValuePath() {
        const numNodes = this.#_nodeValues.length;
        let visited = new Array(numNodes).fill(false);
        let freqMap = new Map();
        let maxValue = 0;

        for (let i = 0; i < numNodes; i++) {
            if (!visited[i]) {
                const pathValue = this.dfs(i, visited, freqMap);

                if (!pathValue) {
                    return null;
                }
                maxValue = Math.max(maxValue, pathValue);
            }
        }

        return maxValue;
    }

    dfs(node, visited, freqMap) {
        visited[node] = true;
        const currNode = this.#_nodeValues[node];
        freqMap.set(currNode, freqMap.get(currNode) + 1 || 1);
        const neighbors = this.#_adjacencyList.get(node) || new Set();
        let maxValue = freqMap.get(currNode);

        for (const neighbor of neighbors) {
            if (visited[neighbor]) {
                return null;
            }

            const pathValue = this.dfs(neighbor, visited, freqMap);
            if (!pathValue) {
                return null;
            }
            maxValue = Math.max(maxValue, pathValue);
        }

        visited[node] = false;
        freqMap.set(currNode, freqMap.get(currNode) - 1);
        if (freqMap.get(currNode) === 0) {
            freqMap.delete(currNode);
        }

        return maxValue;
    }
}

console.log('========= Q21 =========');
const nodes1 = 'ABACA';
const edges1 = [
    [0, 1],
    [0, 2],
    [2, 3],
    [3, 4],
];
const graph1 = new DirectedGraph(nodes1, edges1);
console.log(graph1.findLargestValuePath()); // Output: 3

const nodes2 = 'A';
const edges2 = [[0, 0]];
const graph2 = new DirectedGraph(nodes2, edges2);
console.log(graph2.findLargestValuePath()); // Output: null
console.log('\n');

/*
 * Q22.
 * Given an array of numbers, find the length of the longest increasing
 * subsequence in the array. The subsequence does not necessarily have to be
 * contiguous.
 * For example, given the array [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11,
 * 7, 15], the longest increasing subsequence has length 6: it is 0, 2, 6, 9,
 * 11, 15.
 */
function lengthOfLIS(nums) {
    const n = nums.length;
    if (n === 0) {
        return 0;
    }

    let dp = new Array(n).fill(0);
    dp[0] = 1;
    let maxLength = 1;

    for (let i = 1; i < n; i++) {
        let maxVal = 0;
        for (let j = 0; j < i; j++) {
            if (nums[i] > nums[j]) {
                maxVal = Math.max(maxVal, dp[j]);
            }
        }
        dp[i] = maxVal + 1;
        maxLength = Math.max(maxLength, dp[i]);
    }
    return maxLength;
}

console.log('========= Q22 =========');
const numsToFindLIS = [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15];
console.log(lengthOfLIS(numsToFindLIS)); // Output: 6
console.log('\n');

/*
 * Q23.
 * A rule looks like this:
 * A NE B
 * This means this means point A is located northeast of point B.
 * A SW C
 * means that point A is southwest of C.
 * Given a list of rules, check if the sum of the rules validate. For example:
 * A N B
 * B NE C
 * C N A
 * does not validate, since A cannot be both north and south of C.
 * A NW B
 * A N B
 * is considered valid.
 */
function validateRules(rules) {
    let relationships = new Map();

    for (const rule of rules) {
        const parts = rule.split(' ');
        const point1 = parts[0];
        const direction = parts[1];
        const point2 = parts[2];

        if (!validateRule(relationships, point1, direction, point2)) {
            return false;
        }
    }

    return true;
}

function validateRule(relationships, point1, direction, point2) {
    let existingDirection1 = relationships.get(point1) || new Set();
    let existingDirection2 = relationships.get(point2) || new Set();

    if (
        existingDirection1.has(directionOpposite(direction)) ||
        existingDirection2.has(direction)
    ) {
        return false;
    }

    existingDirection1.add(direction);
    existingDirection2.add(directionOpposite(direction));

    relationships.set(point1, existingDirection1);
    relationships.set(point2, existingDirection2);

    return true;
}

function directionOpposite(direction) {
    switch (direction) {
        case 'N':
            return 'S';
        case 'S':
            return 'N';
        case 'E':
            return 'W';
        case 'W':
            return 'E';
        case 'NE':
            return 'SW';
        case 'SW':
            return 'NE';
        case 'NW':
            return 'SE';
        case 'SE':
            return 'NW';
        default:
            return '';
    }
}

function addRelationship(relationships, point1, direction, point2) {
    if (!relationships.has(point1)) {
        relationships.set(point1, new Set());
    }
    relationships.get(point1).add(direction);

    if (!relationships.has(point2)) {
        relationships.set(point2, new Set());
    }
    relationships.get(point2).add(directionOpposite(direction));
}

console.log('========= Q23 =========');
const rules1 = ['A N B', 'C SE B', 'C N A'];
console.log(validateRules(rules1)); // Output: false

const rules2 = ['A NW B', 'A N B'];
console.log(validateRules(rules2)); // Output: true
console.log('\n');

/*
 * Q24.
 * We're given a hashmap associating each courseId key with a list of courseIds
 * values, which represents that the prerequisites of courseId are courseIds.
 * Return a sorted ordering of courses such that we can finish all courses.
 * Return null if there is no such ordering.
 * For example, given {'CSC300': ['CSC100', 'CSC200'], 'CSC200': ['CSC100'],
 * 'CSC100': []}, should return ['CSC100', 'CSC200', 'CSCS300'].
 */
function findCourseOrder(prerequisites) {
    let graph = new buildGraph(prerequisites);
    let visited = new Set();
    let visiting = new Set();
    let courseOrder = [];

    for (const course of graph.keys()) {
        if (
            !visited.has(course) &&
            !dfs(course, graph, visited, visiting, courseOrder)
        ) {
            return null;
        }
    }

    return courseOrder;
}

function buildGraph(prerequisites) {
    let graph = new Map();

    for (const course of prerequisites.keys()) {
        graph.set(course, []);
    }

    for (const course of prerequisites.keys()) {
        for (const prerequisite of prerequisites.get(course)) {
            graph.get(course).push(prerequisite);
        }
    }

    return graph;
}

function dfs(course, graph, visited, visiting, courseOrder) {
    visiting.add(course);

    for (const prerequisite of graph.get(course)) {
        if (visiting.has(prerequisite)) {
            return false;
        }

        if (
            !visited.has(prerequisite) &&
            !dfs(prerequisite, graph, visited, visiting, courseOrder)
        ) {
            return false;
        }
    }

    visiting.delete(course);
    visited.add(course);
    courseOrder.push(course);
    return true;
}

console.log('========= Q24 =========');
const prerequisites = new Map();
prerequisites.set('CSC300', ['CSC100', 'CSC200']);
prerequisites.set('CSC200', ['CSC100']);
prerequisites.set('CSC100', []);

const courseOrder = findCourseOrder(prerequisites);
if (!courseOrder) {
    console.log('No valid course ordering exists.');
} else {
    console.log(`Course ordering: ${courseOrder}`);
}
console.log('\n');

/*
 * Q25.
 * Given a tree, find the largest tree/subtree that is a BST.
 * Given a tree, return the size of the largest tree/subtree that is a BST.
 */
class TreeNode {
    constructor(val) {
        this.val = val;
        this.left = null;
        this.right = null;
    }
}

class Result {
    constructor(size, min, max, isBST) {
        this.size = size;
        this.min = min;
        this.max = max;
        this.isBST = isBST;
    }
}

function largestBST(root) {
    return largestBSTHelper(root).size;
}

function largestBSTHelper(root) {
    if (!root) {
        return new Result(
            0,
            Number.MAX_SAFE_INTEGER,
            Number.MIN_SAFE_INTEGER,
            true
        );
    }

    const left = largestBSTHelper(root.left);
    const right = largestBSTHelper(root.right);

    if (
        !left.isBST ||
        !right.isBST ||
        root.val < left.max ||
        root.val > right.min
    ) {
        return new Result(Math.max(left.size, right.size), 0, 0, false);
    }

    const size = left.size + right.size + 1;
    const min = root.left ? left.min : root.val;
    const max = root.right ? right.max : root.val;

    return new Result(size, min, max, true);
}

console.log('========= Q25 =========');
const root = new TreeNode(10);
root.left = new TreeNode(5);
root.right = new TreeNode(15);
root.left.left = new TreeNode(1);
root.left.right = new TreeNode(8);
root.right.right = new TreeNode(7);

console.log(`Size of the largest BST: ${largestBST(root)}`); // Output: 3
console.log('\n');

/*
 * Q26.
 * Given a number represented by a list of digits, find the next greater
 * permutation of a number, in terms of lexicographic ordering. If there is not
 * greater permutation possible, return the permutation with the lowest
 * value/ordering.
 * For example, the list [1,2,3] should return [1,3,2]. The list [1,3,2] should
 * return [2,1,3]. The list [3,2,1] should return [1,2,3].
 * Can you perform the operation without allocating extra memory (disregarding
 * the input memory)?
 */
function nextPermutation(nums) {
    const n = nums.length;
    let i = n - 2;

    while (i >= 0 && nums[i] >= nums[i + 1]) {
        i--;
    }

    if (i >= 0) {
        let j = n - 1;
        while (j >= 0 && nums[j] <= nums[i]) {
            j--;
        }
        swap(nums, i, j);
    }

    reverse(nums, i + 1, n - 1);
}

function swap(nums, i, j) {
    const temp = nums[i];
    nums[i] = nums[j];
    nums[j] = temp;
}

function reverse(nums, start, end) {
    while (start < end) {
        swap(nums, start, end);
        start++;
        end--;
    }
}

console.log('========= Q26 =========');
const numsToFindNextPermutation1 = [1, 2, 3];
nextPermutation(numsToFindNextPermutation1);
console.log(`Next permutation: ${numsToFindNextPermutation1}`); // Output: [1, 3, 2]

const numsToFindNextPermutation2 = [1, 3, 2];
nextPermutation(numsToFindNextPermutation2);
console.log(`Next permutation: ${numsToFindNextPermutation2}`); // Output: [2, 1, 3]

const numsToFindNextPermutation3 = [3, 2, 1];
nextPermutation(numsToFindNextPermutation3);
console.log(`Next permutation: ${numsToFindNextPermutation3}`); // Output: [1, 2, 3]
console.log('\n');

/*
 * Q27.
 * Given a word W and a string S, find all starting indices in S which are
 * anagrams of W.
 * For example, given that W is "ab", and S is "abxaba", return 0, 3, and 4.
 */
function findAnagramIndices(s, w) {
    let result = [];

    if (s.length === 0 || w.length === 0 || s.length < w.length) {
        return result;
    }

    let targetFreqMap = new Map();
    let windowFreqMap = new Map();

    for (const ch of w) {
        targetFreqMap.set(ch, (targetFreqMap.get(ch) || 0) + 1);
    }

    const windowSize = w.length;
    for (let i = 0; i < windowSize; i++) {
        const ch = s[i];
        windowFreqMap.set(ch, (windowFreqMap.get(ch) || 0) + 1);
    }

    if (isAnagram(targetFreqMap, windowFreqMap)) {
        result.push(0);
    }

    for (let i = windowSize; i < s.length; i++) {
        const incoming = s[i];
        const outgoing = s[i - windowSize];

        windowFreqMap.set(incoming, (windowFreqMap.get(incoming) || 0) + 1);
        windowFreqMap.set(outgoing, windowFreqMap.get(outgoing) - 1);

        if (windowFreqMap.get(outgoing) === 0) {
            windowFreqMap.delete(outgoing);
        }

        if (isAnagram(targetFreqMap, windowFreqMap)) {
            result.push(i - windowSize + 1);
        }
    }

    return result;
}

function isAnagram(targetFreqMap, windowFreqMap) {
    return (
        targetFreqMap.size === windowFreqMap.size &&
        [...targetFreqMap].every(([ch, freq]) => windowFreqMap.get(ch) === freq)
    );
}

console.log('========= Q27 =========');
const S = 'abxaba';
const W = 'ab';
console.log(`Anagram indices: ${findAnagramIndices(S, W)}`); // Output: [0, 3, 4]

/*
 * Q28.
 * Given a binary tree, find the lowest common ancestor (LCA) of two given nodes
 * in the tree. Assume that each node in the tree also has a pointer to its
 * parent.
 * https://en.wikipedia.org/wiki/Lowest_common_ancestor
 * According to the definition of LCA on Wikipedia: “The lowest common ancestor
 * is defined between two nodes v and w as the lowest node in T that has both v
 * and w as descendants (where we allow a node to be a descendant of itself).”
 */
class TreeNodeWithParent {
    constructor(val) {
        this.val = val;
        this.left = null;
        this.right = null;
        this.parent = null;
    }
}

function findLowestCommonAncestor(root, p, q) {
    if (!root || !p || !q) {
        return null;
    }

    const pathToP = getPathToRoot(p);
    const pathToQ = getPathToRoot(q);

    let i = 0;
    while (i < pathToP.length && i < pathToQ.length) {
        if (pathToP[i] !== pathToQ[i]) {
            break;
        }
        i++;
    }

    if (i > 0) {
        return pathToP[i - 1];
    }

    return null;
}

function getPathToRoot(node) {
    let path = [];

    while (node) {
        path.push(node);
        node = node.parent;
    }

    return path.reverse();
}

console.log('========= Q28 =========');
const treeWithParent = new TreeNodeWithParent(3);
treeWithParent.left = new TreeNodeWithParent(5);
treeWithParent.right = new TreeNodeWithParent(1);

treeWithParent.left.parent = treeWithParent;
treeWithParent.right.parent = treeWithParent;

treeWithParent.left.left = new TreeNodeWithParent(6);
treeWithParent.left.right = new TreeNodeWithParent(2);

treeWithParent.left.left.parent = treeWithParent.left;
treeWithParent.left.right.parent = treeWithParent.left;

treeWithParent.left.right.left = new TreeNodeWithParent(7);
treeWithParent.left.right.right = new TreeNodeWithParent(4);

treeWithParent.left.right.left.parent = treeWithParent.left.right;
treeWithParent.left.right.right.parent = treeWithParent.left.right;

treeWithParent.right.left = new TreeNodeWithParent(0);
treeWithParent.right.right = new TreeNodeWithParent(8);

treeWithParent.right.left.parent = treeWithParent.right;
treeWithParent.right.right.parent = treeWithParent.right;

const p = treeWithParent.left;
const q = treeWithParent.right;

const lca = findLowestCommonAncestor(treeWithParent, p, q);
if (lca) {
    console.log(`Lowest Common Ancestor: ${lca.val}`);
} else {
    console.log(`Lowest Common Ancestor not found`);
}
console.log('\n');

/*
 * Q29.
 * Given a string and a set of delimiters, reverse the words in the string while
 * maintaining the relative order of the delimiters. For example, given
 * "hello/world:here", return "here/world:hello"
 * Follow-up: Does your solution work for the following cases:
 * "hello/world:here/", "hello//world:here"
 */
function reverseWordsWithDelimiters(str, delimiters) {
    let words = str.split(
        new RegExp(`[${getDelimitersAsString(delimiters)}]+`)
    );
    let separators = str.split(
        new RegExp(`[^${getDelimitersAsString(delimiters)}]+`)
    );
    separators = separators.filter((separator) => separator !== '');

    reverseArray(words);

    let reversed = '';
    let i = 0;
    let j = 0;
    while (i < words.length || j < separators.length) {
        if (i < words.length) {
            reversed += words[i++];
        }
        if (j < separators.length) {
            reversed += separators[j++];
        }
    }

    return reversed;
}

function getDelimitersAsString(delimiters) {
    let str = '';
    for (const delimiter of delimiters) {
        str += delimiter;
    }
    return str;
}

function reverseArray(arr) {
    let start = 0;
    let end = arr.length - 1;

    while (start < end) {
        const temp = arr[start];
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    }
}

console.log('========= Q29 =========');
const str1 = 'hello/world:here';
const str2 = 'hello/world:here/';
const str3 = 'hello//world:here';
const delimiters = new Set(['/', ':']);

let reversed = reverseWordsWithDelimiters(str1, delimiters);
console.log(`Reversed string: ${reversed}`);
reversed = reverseWordsWithDelimiters(str2, delimiters);
console.log(`Reversed string: ${reversed}`);
reversed = reverseWordsWithDelimiters(str3, delimiters);
console.log(`Reversed string: ${reversed}`);
console.log('\n');

/*
 * Q30.
 * Given two non-empty binary trees s and t, check whether tree t has exactly
 * the same structure and node values with a subtree of s. A subtree of s is a
 * tree consists of a node in s and all of this node's descendants. The tree s
 * could also be considered as a subtree of itself.
 */
function isSubtree(s, t) {
    if (!s) {
        return false;
    }

    if (isSameTree(s, t)) {
        return true;
    }

    return isSubtree(s.left, t) || isSubtree(s.right, t);
}

function isSameTree(s, t) {
    if (!s && !t) {
        return true;
    }

    if (!s || !t) {
        return false;
    }

    return (
        s.val === t.val &&
        isSameTree(s.left, t.left) &&
        isSameTree(s.right, t.right)
    );
}

console.log('========= Q30 =========');
const binaryTreeS = new TreeNode(3);
binaryTreeS.left = new TreeNode(4);
binaryTreeS.right = new TreeNode(5);
binaryTreeS.left.left = new TreeNode(1);
binaryTreeS.left.right = new TreeNode(2);
binaryTreeS.left.right.left = new TreeNode(0);

const binaryTreeT = new TreeNode(4);
binaryTreeT.left = new TreeNode(1);
binaryTreeT.right = new TreeNode(2);

console.log(`Is t a subtree of s? ${isSubtree(binaryTreeS, binaryTreeT)}`);
console.log('\n');

/*
 * Q31.
 * Given a string which we can delete at most k, return whether you can make a
 * palindrome.
 * For example, given 'waterrfetawx' and a k of 2, you could delete f and x to
 * get 'waterretaw'.
 */
function canMakePalindrome(s, k) {
    const n = s.length;
    let dp = new Array(n).fill(0).map(() => new Array(n).fill(0));

    for (let len = 2; len <= n; len++) {
        for (let i = 0; i < n - len; i++) {
            let j = i + len - 1;

            if (s[i] === s[j]) {
                dp[i][j] = dp[i + 1][j - 1];
            } else {
                dp[i][j] = Math.min(dp[i + 1][j], dp[i][j - 1]) + 1;
            }
        }
    }

    return dp[0][n - 1] <= k;
}

console.log('========= Q31 =========');
const palindromeCandidate = 'waterrfetawx';
const numOfRemoval = 2;
console.log(
    `Can make palindrome: ${canMakePalindrome(
        palindromeCandidate,
        numOfRemoval
    )}`
);
console.log('\n');

/*
 * Q32.
 * Given a string, return whether it represents a number. Here are the different
 * kinds of numbers:
 * "10", a positive integer
 * "-10", a negative integer
 * "10.1", a positive real number
 * "-10.1", a negative real number
 * "1e5", a number in scientific notation
 * And here are examples of non-numbers:
 * "a"
 * "x 1"
 * "a -2"
 * "-"
 */
function isNumber(s) {
    s = s.trim();
    const pattern = /^[-+]?(?:\d+\.?|\.\d+)\d*(?:e[-+]?\d+)?$/;
    return pattern.test(s);
}

console.log('========= Q32 =========');
const inputs = ['10', '-10', '10.1', '-10.1', '1e5', 'a', 'x 1', 'a -2', '-'];
for (const input of inputs) {
    console.log(`${input} is a number: ${isNumber(input)}`);
}
console.log('\n');

/*
 * Q33.
 * Find the minimum number of coins required to make n cents.
 * You can use standard American denominations, that is, 1¢, 5¢, 10¢, and 25¢.
 * For example, given n = 16, return 3 since we can make it with a 10¢, a 5¢,
 * and a 1¢.
 */
function coinChange(n) {
    let coins = [25, 10, 5, 1];

    let dp = new Array(n + 1).fill(n + 1);
    dp[0] = 0;

    for (let i = 1; i <= n; i++) {
        for (const coin of coins) {
            if (coin <= i) {
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
            }
        }
    }
    return dp[n] > n ? -1 : dp[n];
}

console.log('========= Q33 =========');
const sum = 16;
console.log(
    `Minimum number of coins required to make ${sum} cents: ${coinChange(sum)}`
);
console.log('\n');

/*
 * Q34.
 * Implement 3 stacks using a single list:
 * " class Stack:                              "
 * "     def __init__(self):                   "
 * "         self.list = []                    "
 *
 * "     def pop(self, stack_number):          "
 * "         pass                              "
 *
 * "     def push(self, item, stack_number):   "
 * "         pass                              "
 */
class MultiStack {
    #_list;
    #_tops;
    #_stackSize;
    #_numStacks;

    constructor(stackSize, numStacks) {
        this.#_stackSize = stackSize;
        this.#_numStacks = numStacks;
        this.#_list = new Array(stackSize * numStacks);
        this.#_tops = new Array(numStacks).fill(-1);
    }

    push(item, stackNumber) {
        if (this.isFull(stackNumber)) {
            console.log(
                `Stack ${stackNumber} is full. Cannot push item: ${item}`
            );
            return;
        }

        let index = this.getTopIndex(stackNumber);
        index++;
        this.#_list[this.#_stackSize * stackNumber + index] = item;
        this.#_tops[stackNumber] = index;
    }

    pop(stackNumber) {
        if (this.isEmpty(stackNumber)) {
            console.log(`Stack ${stackNumber} is empty. Cannot pop item.`);
            return -1;
        }

        let index = this.getTopIndex(stackNumber);
        const item = this.#_list[this.#_stackSize * stackNumber + index];
        this.#_tops[stackNumber] = index - 1;
        return item;
    }

    isEmpty(stackNumber) {
        return this.#_tops[stackNumber] === -1;
    }

    isFull(stackNumber) {
        return (
            this.#_tops[stackNumber] ===
            this.#_stackSize * (stackNumber + 1) - 1
        );
    }

    getTopIndex(stackNumber) {
        return this.#_tops[stackNumber];
    }
}

console.log('========= Q34 =========');
const stack = new MultiStack(10, 3);
stack.push(1, 0);
stack.push(2, 0);
stack.push(3, 1);
stack.push(4, 1);
stack.push(5, 2);
stack.push(6, 2);

console.log(`Pop from stack 0: ${stack.pop(0)}`);
console.log(`Pop from stack 1: ${stack.pop(1)}`);
console.log(`Pop from stack 2: ${stack.pop(2)}`);
console.log('\n');

/*
 * Q35.
 * You're given a string consisting solely of (, ), and *. * can represent
 * either a (, ), or an empty string. Determine whether the parentheses are
 * balanced.
 * For example, (()* and (*) are balanced. )*( is not balanced.
 */
function isBalanced(str) {
    let minOpen = 0;
    let maxOpen = 0;

    for (const c of str) {
        if (c == '(') {
            minOpen++;
            maxOpen++;
        } else if (c == ')') {
            minOpen = Math.max(minOpen - 1, 0);
            maxOpen--;
        } else {
            minOpen = Math.max(minOpen - 1, 0);
            maxOpen++;
        }

        if (maxOpen < 0) {
            return false; // More closing parentheses encountered than open parentheses
        }
    }

    return minOpen == 0;
}

console.log('========= Q35 =========');
const string1 = '(()*';
const string2 = '(*)';
const string3 = ')*(';

console.log(`Is ${string1} balanced: ${isBalanced(string1)}`);
console.log(`Is ${string2} balanced: ${isBalanced(string2)}`);
console.log(`Is ${string3} balanced: ${isBalanced(string3)}`);
console.log('\n');

/*
 * Q36.
 * Given a list, sort it using this method: reverse(lst, i, j), which reverses
 * lst from i to j.
 */
function sort(lst) {
    const n = lst.length;
    for (let i = 0; i < n; i++) {
        let minIndex = i;
        for (let j = i + 1; j < n; j++) {
            if (lst[j] < lst[minIndex]) {
                minIndex = j;
            }
        }
        reverse(lst, i, minIndex);
    }
}

function reverse(lst, i, j) {
    while (i < j) {
        const temp = lst[i];
        lst[i] = lst[j];
        lst[j] = temp;
        i++;
        j--;
    }
}

console.log('========= Q36 =========');
const lst = [9, 2, 5, 1, 7];
console.log(`Original list: ${lst}`);
sort(lst);
console.log(`Sorted list: ${lst}`);
console.log('\n');

/*
 * Q37.
 * Given a list of numbers L, implement a method sum(i, j) which returns the sum
 * from the sublist L[i:j] (including i, excluding j).
 * For example, given L = [1, 2, 3, 4, 5], sum(1, 3) should return sum([2, 3]),
 * which is 5.
 * You can assume that you can do some pre-processing. sum() should be optimized
 * over the pre-processing step.
 */
class SublistSum {
    #_prefixSums;

    constructor(lst) {
        this.#_prefixSums = [];
        let sum = 0;
        this.#_prefixSums.push(0);
        for (const num of lst) {
            sum += num;
            this.#_prefixSums.push(sum);
        }
    }

    sum(i, j) {
        if (i < 0 || j > this.#_prefixSums.length - 1 || i > j) {
            throw new Error('Invalid sublist range');
        }
        return this.#_prefixSums[j] - this.#_prefixSums[i];
    }
}

console.log('========= Q37 =========');
const L = [1, 2, 3, 4, 5];
const sublistSum = new SublistSum(L);

console.log(`Sum of sublist [1:3]: ${sublistSum.sum(1, 3)}`);
console.log(`Sum of sublist [2:5]: ${sublistSum.sum(2, 5)}`);
console.log(`Sum of sublist [0:5]: ${sublistSum.sum(0, 5)}`);
console.log('\n');

/*
 * Q38.
 * Given a list of points, a central point, and an integer k, find the nearest k
 * points from the central point.
 * For example, given the list of points [(0, 0), (5, 4), (3, 1)], the central
 * point (1, 2), and k = 2, return [(0, 0), (3, 1)].
 */
class Point {
    constructor(x, y) {
        this.x = x;
        this.y = y;
    }
}

function findNearestPoints(points, centralPoint, k) {
    let pq = [];

    for (const point of points) {
        pq.push(point);
        pq.sort(
            (p1, p2) =>
                calculateDistance(p2, centralPoint) -
                calculateDistance(p1, centralPoint)
        );

        if (pq.length > k) {
            pq.shift();
        }
    }

    return pq;
}

function calculateDistance(p1, p2) {
    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

console.log('========= Q38 =========');
const points = [new Point(0, 0), new Point(5, 4), new Point(3, 1)];
const centralPoint = new Point(1, 2);
const numOfPoints = 2;
const nearestPoints = findNearestPoints(points, centralPoint, numOfPoints);
console.log(
    `Nearest ${numOfPoints} points to (${centralPoint.x}: ${nearestPoints.y}): `
);
for (const point of nearestPoints) {
    console.log(`(${point.x}, ${point.y})`);
}
console.log('\n');

/*
 * Q39.
 * Find an efficient algorithm to find the smallest distance (measured in number
 * of words) between any two given words in a string.
 * For example, given words "hello", and "world" and a text content of
 * "dog cat hello cat dog dog hello cat world", return 1 because there's only
 * one word "cat" in between the two words.
 */
function findSmallestDistance(text, word1, word2) {
    const words = text.split(' ');
    let minDistance = Number.MAX_SAFE_INTEGER;
    let prevIndex = -1;

    for (let i = 0; i < words.length; i++) {
        if (words[i] === word1) {
            if (prevIndex !== -1 && i - prevIndex < minDistance) {
                minDistance = i - prevIndex;
            }
            prevIndex = i;
        } else if (words[i] === word2) {
            if (prevIndex !== -1 && i - prevIndex < minDistance) {
                minDistance = i - prevIndex;
            }
            prevIndex = i;
        }
    }
    return minDistance - 1;
}

console.log('========= Q39 =========');
const textToFindWordDistance = 'dog cat hello cat dog dog hello cat world';
const word1 = 'hello';
const word2 = 'world';
console.log(`${findSmallestDistance(textToFindWordDistance, word1, word2)}`);
console.log('\n');

/*
 * Q40.
 * Given a tree where each edge has a weight, compute the length of the longest
 * path in the tree.
 * For example, given the following tree:
 * "    a      "
 * "   /|\     "
 * "  b c d    "
 * "     / \   "
 * "    e   f  "
 * "   / \     "
 * "  g   h    "
 * and the weights: a-b: 3, a-c: 5, a-d: 8, d-e: 2, d-f: 4, e-g: 1, e-h: 1, the
 * longest path would be c -> a -> d -> f, with a length of 17.
 * The path does not have to pass through the root, and each node can have any
 * amount of children.
 */
class GraphNode {
    constructor(id) {
        this.id = id;
        this.edges = [];
    }
}
class TreeNodeEdge {
    constructor(destination, weight) {
        this.destination = destination;
        this.weight = weight;
    }
}

let longestPath = 0;

function calculateLongestPath(root) {
    if (!root) {
        return 0;
    }

    calculatePath(root, null);

    return longestPath;
}

function calculatePath(node, parent) {
    if (node.edges.length === 1 && node !== parent) {
        return 0;
    }

    let maxPath1 = 0;
    let maxPath2 = 0;

    for (const edge of node.edges) {
        if (edge.destination !== parent) {
            const subPath = calculatePath(edge.destination, node) + edge.weight;
            if (subPath > maxPath1) {
                maxPath2 = maxPath1;
                maxPath1 = subPath;
            } else if (subPath > maxPath2) {
                maxPath2 = subPath;
            }
        }
    }
    longestPath = Math.max(longestPath, maxPath1 + maxPath2);
    return maxPath1;
}

console.log('========= Q40 =========');
const a = new GraphNode('a');
const b = new GraphNode('b');
const c = new GraphNode('c');
const d = new GraphNode('d');
const e = new GraphNode('e');
const f = new GraphNode('f');
const g = new GraphNode('g');
const h = new GraphNode('h');

a.edges.push(new TreeNodeEdge(b, 3));
a.edges.push(new TreeNodeEdge(c, 5));
a.edges.push(new TreeNodeEdge(d, 8));

d.edges.push(new TreeNodeEdge(e, 2));
d.edges.push(new TreeNodeEdge(f, 4));

e.edges.push(new TreeNodeEdge(g, 1));
e.edges.push(new TreeNodeEdge(h, 1));

console.log(`Longest Path Length: ${calculateLongestPath(a)}`);
console.log('\n');

/*
 * Q41.
 * https://en.wikipedia.org/wiki/Reverse_Polish_notation
 * Given an arithmetic expression in Reverse Polish Notation, write a program to
 * evaluate it.
 * The expression is given as a list of numbers and operands. For example: [5,
 * 3, '+'] should return 5 + 3 = 8.
 * For example, [15, 7, 1, 1, '+', '-', '/', 3, '*', 2, 1, 1, '+', '+', '-']
 * should return 5, since it is equivalent to ((15 / (7 - (1 + 1))) * 3) - (2 +
 * (1 + 1)) = 5.
 * You can assume the given expression is always valid.
 */
function evaluateRPN(tokens) {
    let stack = [];

    for (const token of tokens) {
        if (isOperator(token)) {
            const operand2 = stack.pop();
            const operand1 = stack.pop();
            const result = performOperation(operand1, operand2, token);
            stack.push(result);
        } else {
            const operand = parseInt(token);
            stack.push(operand);
        }
    }
    return stack.pop();
}

function isOperator(token) {
    return token === '+' || token === '-' || token === '*' || token === '/';
}

function performOperation(operand1, operand2, operator) {
    switch (operator) {
        case '+':
            return operand1 + operand2;
        case '-':
            return operand1 - operand2;
        case '*':
            return operand1 * operand2;
        case '/':
            return operand1 / operand2;
    }
    return 0;
}

console.log('========= Q41 =========');
const expression = [
    '15',
    '7',
    '1',
    '1',
    '+',
    '-',
    '/',
    '3',
    '*',
    '2',
    '1',
    '1',
    '+',
    '+',
    '-',
];
console.log(`Result: ${evaluateRPN(expression)}`);
console.log('\n');

/*
 * Q42.
 * Given a list of words, find all pairs of unique indices such that the
 * concatenation of the two words is a palindrome.
 * For example, given the list ["code", "edoc", "da", "d"], return [(0, 1), (1,
 * 0), (2, 3)].
 */
function findPalindromePairs(words) {
    let pairs = [];

    for (let i = 0; i < words.length; i++) {
        for (let j = 0; j < words.length; j++) {
            if (i !== j && isPalindrome(words[i] + words[j])) {
                let pair = [i, j];
                pairs.push(pair);
            }
        }
    }
    return pairs;
}

function isPalindrome(word) {
    let i = 0;
    let j = word.length - 1;

    while (i < j) {
        if (word[i] !== word[j]) {
            return false;
        }
        i++;
        j--;
    }
    return true;
}

console.log('========= Q42 =========');
const words = ['code', 'edoc', 'da', 'd'];
const pairs = findPalindromePairs(words);
for (const pair of pairs) {
    console.log(`(${pair[0]}, ${pair[1]})`);
}
console.log('\n');

/*
 * Q43.
 * Alice wants to join her school's Probability Student Club. Membership dues
 * are computed via one of two simple probabilistic games.
 * The first game: roll a die repeatedly. Stop rolling once you get a five
 * followed by a six. Your number of rolls is the amount you pay, in dollars.
 * The second game: same, except that the stopping condition is a five followed
 * by a five.
 * Which of the two games should Alice elect to play? Does it even matter? Write
 * a program to simulate the two games and calculate their expected value.
 */
function calculateExpectedValue(target1, target2, numSimulations) {
    let totalRolls = 0;

    for (let i = 0; i < numSimulations; i++) {
        const rolls = simulateGame(target1, target2);
        totalRolls += rolls;
    }

    return totalRolls / numSimulations;
}

function simulateGame(target1, target2) {
    let rolls = 0;
    let target1Found = false;

    while (true) {
        const roll = Math.floor(Math.random() * 6) + 1;
        rolls++;

        if (target1Found && roll === target2) {
            break;
        }

        target1Found = roll === target1;
    }

    return rolls;
}

console.log('========= Q43 =========');
const numSimulations = 1000000;
const game1ExpectedValue = calculateExpectedValue(5, 6, numSimulations);
const game2ExpectedValue = calculateExpectedValue(5, 5, numSimulations);

console.log(`Expected Value for Game 1: ${game1ExpectedValue}`);
console.log(`Expected Value for Game 2: ${game2ExpectedValue}`);

if (game1ExpectedValue < game2ExpectedValue) {
    console.log('Alice should elect to play Game 1.');
} else if (game1ExpectedValue > game2ExpectedValue) {
    console.log('Alice should elect to play Game 2.');
} else {
    console.log(
        'Alice can choose either game; they have the same expected value.'
    );
}
console.log('\n');

/*
 * Q44.
 * Given a string, split it into as few strings as possible such that each
 * string is a palindrome.
 * For example, given the input string racecarannakayak, return ["racecar",
 * "anna", "kayak"].
 * Given the input string abc, return ["a", "b", "c"].
 */
function splitIntoPalindrome(s) {
    let result = [];
    splitIntoPalindromesHelper(s, 0, [], result);
    return result;
}

function splitIntoPalindromesHelper(s, start, current, result) {
    if (start === s.length) {
        result.length = 0;
        result.push(current.slice());
        return;
    }

    for (let i = start + 1; i <= s.length; i++) {
        const substring = s.substring(start, i);
        if (isPalindrome(substring)) {
            current.push(substring);
            splitIntoPalindromesHelper(s, i, current, result);
            current.pop();
        }
    }
}

console.log('========= Q44 =========');
const inputStr1 = 'racecarannakayak';
const splitResult1 = splitIntoPalindrome(inputStr1);
console.log(`Input: ${inputStr1}`);
console.log(`Result: ${splitResult1}`);

const inputStr2 = 'abc';
const splitResult2 = splitIntoPalindrome(inputStr2);
console.log(`Input: ${inputStr2}`);
console.log(`Result: ${splitResult2}`);
console.log('\n');

/*
 * Q45.
 * Describe what happens when you type a URL into your browser and press Enter.
 */
console.log('========= Q45 =========');
/*
 * 1. Browser parses the URL to extract different components of the URL like
 * protocol, host, port, path, query string etc.
 * 2. The browser checks its cache to find the IP address corresponding to the
 * domain name. If not found, it sends a DNS request to a DNS server to obtain
 * the IP address of the server hosting the website.
 * 3. The browser initiates a TCP connection with the server using the obtained
 * IP address and the default port for the protocol.
 * 4. The browser sends a HTTP request to the server, including the requested
 * path, query parameters, headers, and any additional data required.
 * 5. The server receives the HTTP request and processes it. This may involve
 * executing server-side scripts, accessing databases, or performing other
 * operations to generate a response.
 * 6. The server generates an HTTP response containing the requested content,
 * along with status code, headers, and any additional data.
 * 7. The browser receives the HTTP response and starts rendering the webpage.
 * It interprets the HTML, CSS, and JavaScript code to construct the visual
 * layout, apply styles, and execute any scripts.
 * 8. As the browser parses the HTML, it encounters additional resources such as
 * images, stylesheets, or scripts, referenced in the webpage. It sends separate
 * requests for each resource and starts downloading them in parallel.
 * 9. Once all the resources are downloaded and processed, the browser displays
 * the fully rendered webpage to the user, including text, image, and
 * interactive elements.
 * 10. The user can now interact with the webpage by clicking on links,
 * submitting forms, or performing other actions, which trigger additional
 * requests and responses.
 */
console.log('\n');

/*
 * Q46.
 * Given an array of positive integers, divide the array into two subsets such
 * that the difference between the sum of the subsets is as small as possible.
 * For example, given [5, 10, 15, 20, 25], return the sets {10, 25} and {5, 15,
 * 20}, which has a difference of 5, which is the smallest possible difference.
 */
function minSubsetSumDifference(nums) {
    let totalSum = 0;
    for (const num of nums) {
        totalSum += num;
    }

    let dp = new Array(nums.length + 1)
        .fill(false)
        .map(() => new Array(totalSum + 1).fill(false));

    for (let i = 0; i <= nums.length; i++) {
        dp[i][0] = true;
    }

    for (let i = 1; i <= nums.length; i++) {
        for (let j = 1; j <= totalSum; j++) {
            if (nums[i - 1] <= j) {
                dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]];
            } else {
                dp[i][j] = dp[i - 1][j];
            }
        }
    }

    let minDiff = Number.MAX_SAFE_INTEGER;

    for (let j = Math.floor(totalSum / 2); j >= 0; j--) {
        if (dp[nums.length][j]) {
            minDiff = totalSum - 2 * j;
            break;
        }
    }

    printSubsets(nums, dp, (totalSum - minDiff) / 2);

    return minDiff;
}

function printSubsets(nums, dp, median) {
    let subset1 = [];
    let subset2 = [];
    let i = nums.length;
    let j = median;

    while (i > 0 && j > 0) {
        if (nums[i - 1] <= j && dp[i - 1][j - nums[i - 1]]) {
            subset1.push(nums[i - 1]);
            j -= nums[i - 1];
        }
        i--;
    }

    for (const num of nums) {
        if (!subset1.includes(num)) {
            subset2.push(num);
        }
    }

    console.log(`Subset 1: ${subset1}`);
    console.log(`Subset 2: ${subset2}`);
}

console.log('========= Q46 =========');
const numsToDivide = [5, 10, 15, 20, 25];
const minDiff = minSubsetSumDifference(numsToDivide);
console.log(`Minimum subset sum difference: ${minDiff}`);
console.log('\n');

/*
 * Q47.
 * Given a array of numbers representing the stock prices of a company in
 * chronological order, write a function that calculates the maximum profit you
 * could have made from buying and selling that stock. You're also given a
 * number fee that represents a transaction fee for each buy and sell
 * transaction.
 * You must buy before you can sell the stock, but you can make as many
 * transactions as you like.
 * For example, given [1, 3, 2, 8, 4, 10] and fee = 2, you should return 9,
 * since you could buy the stock at 1 dollar, and sell at 8 dollars, and then
 * buy it at 4 dollars and sell it at 10 dollars. Since we did two transactions,
 * there is a 4 dollar fee, so we have 7 + 6 = 13 profit minus 4 dollars of
 * fees.
 */
function maxProfit(prices, fee) {
    const n = prices.length;
    let dp = new Array(n).fill(0).map(() => new Array(2).fill(0));

    dp[0][0] = 0;
    dp[0][1] = -prices[0];

    for (let i = 1; i < n; i++) {
        dp[i][0] = Math.max(dp[i - 1][0], dp[i - 1][1] + prices[i] - fee);
        dp[i][1] = Math.max(dp[i - 1][1], dp[i - 1][0] - prices[i]);
    }

    return dp[n - 1][0];
}

console.log('========= Q47 =========');
const prices = [1, 3, 2, 8, 4, 10];
const fee = 2;
const maxProfitWithFee = maxProfit(prices, fee);
console.log(`Maximum profit: ${maxProfitWithFee}`);
console.log('\n');

/*
 * Q48.
 * Let A be an N by M matrix in which every row and every column is sorted.
 * Given i1, j1, i2, and j2, compute the number of elements of M smaller than
 * M[i1, j1] and larger than M[i2, j2].
 * For example, given the following matrix:
 * [[1, 3, 7, 10, 15, 20],
 * [2, 6, 9, 14, 22, 25],
 * [3, 8, 10, 15, 25, 30],
 * [10, 11, 12, 23, 30, 35],
 * [20, 25, 30, 35, 40, 45]]
 * And i1 = 1, j1 = 1, i2 = 3, j2 = 3, return 15 as there are 15 numbers in the
 * matrix smaller than 6 or greater than 23.
 */
function countElements(matrix, i1, j1, i2, j2) {
    let count = 0;
    const num1 = matrix[i1][j1];
    const num2 = matrix[i2][j2];

    for (const row of matrix) {
        for (const num of row) {
            if (num < num1 || num > num2) {
                count++;
            }
        }
    }

    return count;
}

console.log('========= Q48 =========');
const matrix = [
    [1, 3, 7, 10, 15, 20],
    [2, 6, 9, 14, 22, 25],
    [3, 8, 10, 15, 25, 30],
    [10, 11, 12, 23, 30, 35],
    [20, 25, 30, 35, 40, 45],
];
const i1 = 1;
const j1 = 1;
const i2 = 3;
const j2 = 3;
const elementCount = countElements(matrix, i1, j1, i2, j2);
console.log(`Number of elements: ${elementCount}`);
console.log('\n');

/*
 * Q49.
 * Given a string of parentheses, find the balanced string that can be produced
 * from it using the minimum number of insertions and deletions. If there are
 * multiple solutions, return any of them.
 * For example, given "(()", you could return "(())". Given "))()(", you could
 * return "()()()()".
 */
function balanceParentheses(s) {
    let balanceString = '';
    let stack = [];

    for (const c of s) {
        if (c === '(') {
            stack.push(c);
            balanceString += c;
        } else if (c === ')') {
            if (stack.length > 0 && stack[stack.length - 1] === '(') {
                stack.pop();
                balanceString += c;
            } else {
                balanceString += '(' + c;
            }
        } else {
            balanceString += c;
        }
    }

    while (stack.length > 0) {
        balanceString += ')';
        stack.pop();
    }

    return balanceString;
}

console.log('========= Q49 =========');
const parentheses1 = '(()';
const parentheses2 = '))()(';

console.log(
    `Balanced string for ${parentheses1}: ${balanceParentheses(parentheses1)}`
);
console.log(
    `Balanced string for ${parentheses2}: ${balanceParentheses(parentheses2)}`
);
console.log('\n');

/*
 * Q50.
 * Let X be a set of n intervals on the real line. We say that a set of points P
 * "stabs" X if every interval in X contains at least one point in P. Compute
 * the smallest set of points that stabs X.
 * For example, given the intervals [(1, 4), (4, 5), (7, 9), (9, 12)], you
 * should return [4, 9].
 */
class Interval {
    constructor(start, end) {
        this.start = start;
        this.end = end;
    }
}

function findStabPoints(intervals) {
    intervals.sort((a, b) => a.end - b.end);
    let points = [];
    let currentPoint = intervals[0].end;

    for (const interval of intervals) {
        if (interval.start > currentPoint) {
            points.push(currentPoint);
            currentPoint = interval.end;
        }
    }

    points.push(currentPoint);
    return points;
}

console.log('========= Q50 =========');
const intervals = [
    new Interval(1, 4),
    new Interval(4, 5),
    new Interval(7, 9),
    new Interval(9, 12),
];
const stabPoints = findStabPoints(intervals);
console.log(`Smallest set of stab points: ${stabPoints}`);
console.log('\n');

/*
 * Q51.
 * Write a program that computes the length of the longest common subsequence of
 * three given strings. For example, given "epidemiologist", "refrigeration",
 * and "supercalifragilisticexpialodocious", it should return 5, since the
 * longest common subsequence is "eieio".
 */
function longestCommonSubsequenceLength(text1, text2, text3) {
    const m = text1.length;
    const n = text2.length;
    const p = text3.length;

    let dp = new Array(m + 1)
        .fill(0)
        .map(() =>
            new Array(n + 1).fill(0).map(() => new Array(p + 1).fill(0))
        );

    for (let i = 1; i <= m; i++) {
        for (let j = 1; j <= n; j++) {
            for (let k = 1; k <= p; k++) {
                if (
                    text1[i - 1] === text2[j - 1] &&
                    text1[i - 1] === text3[k - 1]
                ) {
                    dp[i][j][k] = dp[i - 1][j - 1][k - 1] + 1;
                } else {
                    dp[i][j][k] = Math.max(
                        dp[i - 1][j][k],
                        dp[i][j - 1][k],
                        dp[i][j][k - 1]
                    );
                }
            }
        }
    }
    return dp[m][n][p];
}

console.log('========= Q51 =========');
const text1 = 'epidemiologist';
const text2 = 'refrigeration';
const text3 = 'supercalifragilisticexpialodocious';
const commonSubstringLength = longestCommonSubsequenceLength(
    text1,
    text2,
    text3
);
console.log(`Length of longest common subsequence: ${commonSubstringLength}`);
console.log('\n');

/*
 * Q52.
 * We say a number is sparse if there are no adjacent ones in its binary
 * representation. For example, 21 (10101) is sparse, but 22 (10110) is not. For
 * a given input N, find the smallest sparse number greater than or equal to N.
 * Do this in faster than O(N log N) time.
 */
function findSparseNumber(N) {
    const binary = N.toString(2);
    const binaryArray = binary.split('');

    for (let i = 1; i < binaryArray.length; i++) {
        if (binaryArray[i] === '1' && binaryArray[i - 1] === '1') {
            for (let j = i; j < binaryArray.length; j = j + 2) {
                binaryArray[j] = '0';
                if (j + 1 < binaryArray.length) {
                    binaryArray[j + 1] = '1';
                }
            }
            return parseInt(binaryArray.join(''), 2);
        }
    }
    return N;
}

console.log('========= Q52 =========');
const nonSparseNum = 22;
const sparseNum = findSparseNumber(nonSparseNum);
console.log(
    `Smallest sparse number greater than or equal to ${nonSparseNum}: ${sparseNum}`
);
console.log('\n');

/*
 * Q53.
 * Connect 4 is a game where opponents take turns dropping red or black discs
 * into a 7 x 6 vertically suspended grid. The game ends either when one player
 * creates a line of four consecutive discs of their color (horizontally,
 * vertically, or diagonally), or when there are no more spots left in the grid.
 * Design and implement Connect 4.
 */
const readline = require('readline');
class Connect4 {
    #_ROWS = 6;
    #_COLUMNS = 7;
    #_EMPTY = '-';
    #_RED = 'R';
    #_BLACK = 'B';
    #_board = [];
    #_currentPlayer;

    initializeGame() {
        this.#_board = new Array(this.#_ROWS)
            .fill(this.#_EMPTY)
            .map(() => new Array(this.#_COLUMNS).fill(this.#_EMPTY));
        this.#_currentPlayer = this.#_RED;
    }

    async playGame() {
        let gameEnded = false;

        while (!gameEnded) {
            this.#_displayBoard();
            console.log(`Player ${this.#_currentPlayer}'s turn`);
            const column = parseInt(await this.#_getPlayerMove());

            if (
                this.#_columnIsValid(column) &&
                this.#_columnIsNotFull(column)
            ) {
                const row = this.#_dropDisc(column);
                if (this.#_checkWin(row, column)) {
                    this.#_displayBoard();
                    console.log(`Player ${this.#_currentPlayer} wins!`);
                    gameEnded = true;
                } else if (this.#_boardIsFull()) {
                    this.#_displayBoard();
                    console.log("It's a draw!");
                    gameEnded = true;
                } else {
                    this.#_switchPlayers();
                }
            } else {
                console.log('Invalid move. Please try again.');
            }
        }
    }

    #_displayBoard() {
        let printBoard = '';
        for (let i = this.#_ROWS - 1; i >= 0; i--) {
            for (let j = 0; j < this.#_COLUMNS; j++) {
                printBoard += this.#_board[i][j] + ' ';
            }
            printBoard += '\n';
        }
        console.log(printBoard + '\n');
    }

    async #_getPlayerMove() {
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout,
        });

        return new Promise((resolve) => {
            rl.question('Enter a column (0-6): ', (userInput) => {
                rl.close();
                resolve(userInput);
            });
        });
    }

    #_columnIsValid(column) {
        return column >= 0 && column < this.#_COLUMNS;
    }

    #_columnIsNotFull(column) {
        return this.#_board[this.#_ROWS - 1][column] === this.#_EMPTY;
    }

    #_dropDisc(column) {
        let row = 0;
        while (row < this.#_ROWS) {
            if (this.#_board[row][column] === this.#_EMPTY) {
                break;
            }
            row++;
        }
        this.#_board[row][column] = this.#_currentPlayer;
        return row;
    }

    #_checkWin(row, column) {
        return (
            this.#_checkHorizontal(row) ||
            this.#_checkVertical(column) ||
            this.#_checkDiagonal(row, column)
        );
    }

    #_checkHorizontal(row) {
        let count = 0;
        for (let i = 0; i < this.#_COLUMNS; i++) {
            if (this.#_board[row][i] === this.#_currentPlayer) {
                count++;
                if (count === 4) {
                    return true;
                }
            } else {
                count = 0;
            }
        }
        return false;
    }

    #_checkVertical(column) {
        let count = 0;
        for (let i = 0; i < this.#_ROWS; i++) {
            if (this.#_board[i][column] === this.#_currentPlayer) {
                count++;
                if (count === 4) {
                    return true;
                }
            } else {
                count = 0;
            }
        }
        return false;
    }

    #_checkDiagonal(row, column) {
        let count = 0;
        let i = row;
        let j = column;
        while (
            i >= 0 &&
            j < this.#_COLUMNS &&
            this.#_board[i][j] === this.#_currentPlayer
        ) {
            count++;
            if (count === 4) {
                return true;
            }
            i--;
            j++;
        }

        i = row;
        j = column;
        while (
            i >= 0 &&
            j >= 0 &&
            this.#_board[i][j] === this.#_currentPlayer
        ) {
            count++;
            if (count === 4) {
                return true;
            }
            i--;
            j--;
        }

        return false;
    }

    #_boardIsFull() {
        for (let i = 0; i < this.#_ROWS; i++) {
            for (let j = 0; j < this.#_COLUMNS; j++) {
                if (this.#_board[i][j] === this.#_EMPTY) {
                    return false;
                }
            }
        }
        return true;
    }

    #_switchPlayers() {
        this.#_currentPlayer =
            this.#_currentPlayer === this.#_RED ? this.#_BLACK : this.#_RED;
    }
}

console.log('========= Q53 =========');
const connect4 = new Connect4();
connect4.initializeGame();
connect4.playGame();
console.log('\n');

/*
 * Q54.
 * Typically, an implementation of in-order traversal of a binary tree has O(h)
 * space complexity, where h is the height of the tree. Write a program to
 * compute the in-order traversal of a binary tree using O(1) space.
 */
function morrisTraversal(root) {
    let current = root;
    while (current) {
        if (!current.left) {
            console.log(current.val);
            current = current.right;
        } else {
            // Find the rightmost node in the left subtree
            let predecessor = current.left;
            while (predecessor.right && predecessor.right !== current) {
                predecessor = predecessor.right;
            }

            if (!predecessor.right) {
                // Make current the right child of its inorder predecessor
                predecessor.right = current;
                current = current.left;
            } else {
                predecessor.right = null; // Restore the original tree structure
                console.log(current.val);
                current = current.right;
            }
        }
    }
}

console.log('========= Q54 =========');
const rootForInorderTraversal = new TreeNode(1);
rootForInorderTraversal.left = new TreeNode(2);
rootForInorderTraversal.right = new TreeNode(3);
rootForInorderTraversal.left.left = new TreeNode(4);
rootForInorderTraversal.left.right = new TreeNode(5);

morrisTraversal(rootForInorderTraversal);
console.log('\n');

/*
 * Q55.
 * You come across a dictionary of sorted words in a language you've never seen
 * before. Write a program that returns the correct order of letters in this
 * language.
 * For example, given ['xww', 'wxyz', 'wxyw', 'ywx', 'ywz'], you should return
 * ['x', 'z', 'w', 'y'].
 */
function getLanguageOrder(words) {
    let graph = new Map();

    for (const word of words) {
        for (const c of word) {
            graph.set(c, new Set());
        }
    }

    for (let i = 1; i < words.length; i++) {
        const prevWord = words[i - 1];
        const currWord = words[i];
        const minLength = Math.min(prevWord.length, currWord.length);

        for (let j = 0; j < minLength; j++) {
            const prevChar = prevWord[j];
            const currChar = currWord[j];

            if (prevChar !== currChar) {
                graph.get(prevChar).add(currChar);
            }
        }
    }

    let order = [];
    let visited = new Set();

    for (const c of graph.keys()) {
        if (!visited.has(c)) {
            dfsLetters(graph, c, visited, order);
        }
    }

    return order;
}

function dfsLetters(graph, c, visited, order) {
    visited.add(c);

    for (const neighbor of graph.get(c)) {
        if (!visited.has(neighbor)) {
            dfsLetters(graph, neighbor, visited, order);
        }
    }

    order.unshift(c);
}

console.log('========= Q55 =========');
const wordsFromNewLang = ['xww', 'wxyz', 'wxyw', 'ywx', 'ywz'];
const order = getLanguageOrder(wordsFromNewLang);
console.log('Correct order of letters in the language:');
for (const ch of order) {
    console.log(ch);
}
console.log('\n');

/*
 * Q56.
 * Recall that the minimum spanning tree is the subset of edges of a tree that
 * connect all its vertices with the smallest possible total edge weight. Given
 * an undirected graph with weighted edges, compute the maximum weight spanning
 * tree.
 */
class Edge {
    constructor(src, dest, weight) {
        this.src = src;
        this.dest = dest;
        this.weight = weight;
    }
}

class UnionFind {
    constructor(size) {
        this.parent = new Array(size);
        this.rank = new Array(size);
        for (let i = 0; i < size; i++) {
            this.parent[i] = i;
            this.rank[i] = 0;
        }
    }

    find(x) {
        if (this.parent[x] !== x) {
            this.parent[x] = this.find(this.parent[x]);
        }
        return this.parent[x];
    }

    union(x, y) {
        const rootX = this.find(x);
        const rootY = this.find(y);

        if (this.rank[rootX] < this.rank[rootY]) {
            this.parent[rootX] = rootY;
        } else if (this.rank[rootX] > this.rank[rootY]) {
            this.parent[rootY] = rootX;
        } else {
            this.parent[rootY] = rootX;
            this.rank[rootX]++;
        }
    }
}

function findMaximumSpanningTree(edges, numVertices) {
    edges.sort((a, b) => b.weight - a.weight);

    let uf = new UnionFind(numVertices);
    let mst = [];

    for (const edge of edges) {
        const srcParent = uf.find(edge.src);
        const destParent = uf.find(edge.dest);

        if (srcParent !== destParent) {
            uf.union(srcParent, destParent);
            mst.push(edge);
        }
    }
    return mst;
}

console.log('========= Q56 =========');
const edges = [];
edges.push(new Edge(0, 1, 10));
edges.push(new Edge(0, 2, 6));
edges.push(new Edge(0, 3, 5));
edges.push(new Edge(1, 3, 15));
edges.push(new Edge(2, 3, 4));

const numVertices = 4;
const mst = findMaximumSpanningTree(edges, numVertices);

console.log('Edges of the maximum weight spanning tree:');
for (const edge of mst) {
    console.log(`${edge.src} -- ${edge.dest} : ${edge.weight}`);
}
console.log('\n');

/*
 * Q57.
 * Given an array of numbers of length N, find both the minimum and maximum
 * using less than 2 * (N - 2) comparisons.
 */
class MinAndMax {
    constructor(min, max) {
        this.min = min;
        this.max = max;
    }
}

function findMinMax(arr) {
    if (arr.length === 0) {
        throw new Error('Array must not be empty');
    }

    let min, max;

    if (arr[0] < arr[1]) {
        min = arr[0];
        max = arr[1];
    } else {
        min = arr[1];
        max = arr[0];
    }

    for (let i = 2; i < arr.length - 1; i += 2) {
        const num1 = arr[i];
        const num2 = arr[i + 1];

        if (num1 < num2) {
            min = Math.min(min, num1);
            max = Math.max(max, num2);
        } else {
            min = Math.min(min, num2);
            max = Math.max(max, num1);
        }
    }

    if (arr.length % 2 === 1) {
        const lastNum = arr[arr.length - 1];
        min = Math.min(min, lastNum);
        max = Math.max(max, lastNum);
    }

    return new MinAndMax(min, max);
}

console.log('========= Q57 =========');
const arr = [5, 7, 1, 3, 9, 2];
const minAndMax = findMinMax(arr);
console.log(`Minimum: ${minAndMax.min}`);
console.log(`Maximum: ${minAndMax.max}`);
console.log('\n');

/*
 * Q58.
 * https://en.wikipedia.org/wiki/Blackjack
 * Blackjack is a two player card game whose rules are as follows:
 * The player and then the dealer are each given two cards.
 * The player can then "hit", or ask for arbitrarily many additional cards, so
 * long as their total does not exceed 21.
 * The dealer must then hit if their total is 16 or lower, otherwise pass.
 * Finally, the two compare totals, and the one with the greatest sum not
 * exceeding 21 is the winner.
 * For this problem, cards values are counted as follows: each card between 2
 * and 10 counts as their face value, face cards count as 10, and aces count as
 * 1.
 * Given perfect knowledge of the sequence of cards in the deck, implement a
 * blackjack solver that maximizes the player's score (that is, wins minus
 * losses).
 */
function solveBlackjack(deck) {
    let wins = 0;
    let losses = 0;

    const iterations = 10000;
    for (let i = 0; i < iterations; i++) {
        const playerScore = playBlackjack(deck);
        const dealerScore = playBlackjack(deck);

        if (
            playerScore <= 21 &&
            (playerScore > dealerScore || dealerScore > 21)
        ) {
            wins++;
        } else if (
            playerScore > 21 ||
            (playerScore < dealerScore && dealerScore <= 21)
        ) {
            losses++;
        } else if (playerScore === dealerScore) {
            continue;
        }
    }
    return wins - losses;
}

function playBlackjack(deck) {
    for (let i = deck.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [deck[i], deck[j]] = [deck[j], deck[i]];
    }
    let score = 0;
    let numAces = 0;

    for (const card of deck) {
        if (card >= 2 && card <= 10) {
            score += card;
        } else if (card >= 11 && card <= 13) {
            score += 10;
        } else if (card == 1) {
            score += 1;
            numAces++;
        }

        while (numAces > 0 && score <= 11) {
            score += 10;
            numAces--;
        }

        if (score >= 13 || score >= 21) {
            break;
        }
    }

    return score;
}

console.log('========= Q58 =========');
let deck = [];
for (let i = 1; i <= 13; i++) {
    deck.push(i);
}

const score = solveBlackjack(deck);
console.log(`Score: ${score}`);
console.log('\n');

/*
 * Q59.
 * There are N couples sitting in a row of length 2 * N. They are currently
 * ordered randomly, but would like to rearrange themselves so that each
 * couple's partners can sit side by side.
 * What is the minimum number of swaps necessary for this to happen?
 */
function minSwaps(row) {
    const n = row.length;
    let swaps = 0;

    for (let i = 0; i < n; i++) {
        const partner = row[i] ^ 1;
        if (partner % 2 == 1 && row[i + 1] !== partner) {
            const j = findPartnerIndex(row, i + 1, partner);
            swap(row, i + 1, j);
            swaps++;
        } else if (partner % 2 == 0 && row[i - 1] !== partner) {
            const j = findPartnerIndex(row, i + 1, partner);
            swap(row, i - 1, j);
            swaps++;
        }
    }
    return swaps;
}

function findPartnerIndex(row, start, partner) {
    for (let i = start; i < row.length; i++) {
        if (row[i] === partner) {
            return i;
        }
    }
    return -1;
}

console.log('========= Q59 =========');
const row = [0, 3, 2, 1, 4, 7, 6, 5];
console.log(`Swaps: ${minSwaps(row)}`);
console.log('\n');

/*
 * Q60.
 * You are given an array of length 24, where each element represents the number
 * of new subscribers during the corresponding hour. Implement a data structure
 * that efficiently supports the following:
 * update(hour: int, value: int): Increment the element at index hour by value.
 * query(start: int, end: int): Retrieve the number of subscribers that have
 * signed up between start and end (inclusive).
 * You can assume that all values get cleared at the end of the day, and that
 * you will not be asked for start and end values that wrap around midnight.
 */
class SubscriberTracker {
    #_prefixSum;
    constructor(subscribers) {
        this.#_prefixSum = new Array(subscribers.length + 1).fill(0);
        this.buildPrefixSum(subscribers);
    }

    buildPrefixSum(subscribers) {
        this.#_prefixSum[0] = 0;
        for (let i = 1; i <= subscribers.length; i++) {
            this.#_prefixSum[i] = this.#_prefixSum[i - 1] + subscribers[i - 1];
        }
    }

    update(hour, value) {
        if (hour >= 1 && hour <= this.#_prefixSum.length) {
            while (hour < this.#_prefixSum.length) {
                this.#_prefixSum[hour] += value;
                hour++;
            }
        }
    }

    query(start, end) {
        if (start > 0) {
            return this.#_prefixSum[end] - this.#_prefixSum[start - 1];
        } else {
            return this.#_prefixSum[end];
        }
    }
}

console.log('========= Q60 =========');
const subscribers = [
    5, 3, 7, 2, 8, 4, 10, 6, 15, 9, 11, 5, 14, 7, 13, 8, 12, 6, 9, 10, 7, 11, 5,
    4,
];
const tracker = new SubscriberTracker(subscribers);
tracker.update(5, 10);
tracker.update(10, 5);
tracker.update(15, 8);

console.log(tracker.query(1, 24)); // Output: 214
console.log(tracker.query(5, 15)); // Output: 125
console.log(tracker.query(10, 18)); // Output: 98
console.log('\n');


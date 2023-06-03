/*
 * Q1.
 * Given the root to a binary tree, implement serialize(root), which serializes
 * the tree into a string, and deserialize(s), which deserializes the string
 * back into the tree.
 * For example, given the following Node class
 * "class Node:                                            "
 * "    def __init__(self, val, left=None, right=None):    "
 * "        self.val = val                                 "
 * "        self.left = left                               "
 * "        self.right = right                             "
 * The following test should pass:
 * node = Node('root', Node('left', Node('left.left')), Node('right'))
 * assert deserialize(serialize(node)).left.left.val == 'left.left'
 */
class Node {
    constructor(val, left = null, right = null) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

function serialize(root) {
    if (!root) {
        return 'null';
    }

    const leftSerialized = serialize(root.left);
    const rightSerialized = serialize(root.right);

    return `${root.val},${leftSerialized},${rightSerialized}`;
}

function deserialize(data) {
    const nodes = data.split(',');
    return deserializeHelper(nodes);
}

function deserializeHelper(nodes) {
    const val = nodes.shift();
    if (val === 'null') {
        return null;
    }

    const node = new Node(val);
    node.left = deserializeHelper(nodes);
    node.right = deserializeHelper(nodes);

    return node;
}

console.log('========= Q1 =========');
const node = new Node(
    'root',
    new Node('left', new Node('left.left')),
    new Node('right')
);
const serialized = serialize(node);
console.log(`Serialized Tree: ${serialized}`);
const deserialized = deserialize(serialized);
console.log(`Deserialized Tree: ${deserialized.left.left.val}`);
console.log('\n');

/*
 * Q2.
 * cons(a, b) constructs a pair, and car(pair) and cdr(pair) returns the first
 * and last element of that pair. For example, car(cons(3, 4)) returns 3, and
 * cdr(cons(3, 4)) returns 4.
 * Given this implementation of cons:
 * "def cons(a, b):        "
 * "    def pair(f):       "
 * "        return f(a, b) "
 * "    return pair        "
 * Implement car and cdr.
 */
class Pair {
    constructor(first, second) {
        this.first = first;
        this.second = second;
    }

    car() {
        return this.first;
    }

    cdr() {
        return this.second;
    }
}

function cons(a, b) {
    return new Pair(a, b);
}

function car(pair) {
    return pair.car();
}

function cdr(pair) {
    return pair.cdr();
}

console.log('========= Q2 =========');
const pair = cons(3, 4);
console.log(`car(cons(3, 4)) = ${car(pair)}`);
console.log(`cdr(cons(3, 4)) = ${cdr(pair)}`);
console.log('\n');

/*
 * Q3.
 * Given the mapping a = 1, b = 2, ... z = 26, and an encoded message, count the
 * number of ways it can be decoded.
 * For example, the message '111' would give 3, since it could be decoded as
 * 'aaa', 'ka', and 'ak'.
 * You can assume that the messages are decodable. For example, '001' is not
 * allowed.
 */
function numDecodings(message) {
    if (!message || message.length === 0) {
        return 0;
    }

    const n = message.length;
    let dp = new Array(n + 1).fill(0);
    dp[0] = 1;

    // Handle single-digit cases
    dp[1] = message[0] === '0' ? 0 : 1;

    for (let i = 2; i <= n; i++) {
        const oneDigit = parseInt(message.substring(i - 1, i));
        const twoDigits = parseInt(message.substring(i - 2, i));

        if (oneDigit >= 1 && oneDigit <= 9) {
            dp[i] += dp[i - 1];
        }

        if (twoDigits >= 10 && twoDigits <= 26) {
            dp[i] += dp[i - 2];
        }
    }

    return dp[n];
}

console.log('========= Q3 =========');
const message = '111';
console.log(`Number of ways to decode ${message}: ${numDecodings(message)}`);
console.log('\n');

/*
 * Q4.
 * Implement a job scheduler which takes in a function f and an integer n, and
 * calls f after n milliseconds.
 */
function scheduleJob(task, delayMillis) {
    setTimeout(task, delayMillis);
}

function myTask() {
    console.log('Job executed after delay');
}

console.log('========= Q4 =========');
const delayMillis = 3000;
scheduleJob(myTask, delayMillis);
console.log('Job scheduled');
console.log('\n');

/*
 * Q5.
 * Implement an autocomplete system. That is, given a query string s and a set
 * of all possible query strings, return all strings in the set that have s as a
 * prefix.
 * For example, given the query string de and the set of strings [dog, deer,
 * deal], return [deer, deal].
 * Hint: Try preprocessing the dictionary into a more efficient data structure
 * to speed up queries.
 */
class TrieNode {
    constructor() {
        this.children = Array(26).fill(null);
        // .map(() => new TrieNode());
        this._isEndOfWord = false;
    }

    getChild(ch) {
        return this.children[ch.charCodeAt(0) - 'a'.charCodeAt(0)];
    }

    setChild(ch, node) {
        this.children[ch.charCodeAt(0) - 'a'.charCodeAt(0)] = node;
    }

    get isEndOfWord() {
        return this._isEndOfWord;
    }

    set isEndOfWord(endOfWord) {
        this._isEndOfWord = endOfWord;
    }
}

class AutocompleteSystem {
    #root;

    constructor() {
        this.#root = new TrieNode();
    }

    insert(word) {
        let current = this.#root;

        for (const ch of word) {
            let child = current.getChild(ch);

            if (!child) {
                child = new TrieNode();
                current.setChild(ch, child);
            }
            current = child;
        }
        current.isEndOfWord = true;
    }

    search(prefix) {
        let results = [];
        let current = this.#root;

        for (const ch of prefix) {
            let child = current.getChild(ch);
            if (!child) {
                return results;
            }
            current = child;
        }

        this.collectWords(current, prefix, results);
        return results;
    }

    collectWords(node, prefix, results) {
        if (node.isEndOfWord) {
            results.push(prefix);
        }

        for (let ch = 'a'.charCodeAt(0); ch <= 'z'.charCodeAt(0); ch++) {
            const currentChar = String.fromCharCode(ch);
            let child = node.getChild(currentChar);
            if (child) {
                this.collectWords(child, prefix + currentChar, results);
            }
        }
    }
}

console.log('========= Q5 =========');
const autocomplete = new AutocompleteSystem();
autocomplete.insert('dog');
autocomplete.insert('deer');
autocomplete.insert('deal');

const query = 'de';
const autocompleteResults = autocomplete.search(query);
console.log(`Autocomplete results for ${query}: ${autocompleteResults}`);
console.log('\n');

/*
 * Q6.
 * The area of a circle is defined as πr^2. Estimate π to 3 decimal places using
 * a Monte Carlo method.
 * Hint: The basic equation of a circle is x2 + y2 = r2.
 */
function estimatePieDecimalPlaces() {
    const totalPoints = 1000000;
    let pointsInsideCircle = 0;

    for (let i = 0; i < totalPoints; i++) {
        const x = Math.random();
        const y = Math.random();

        const distance = x * x + y * y;
        if (distance <= 1) {
            pointsInsideCircle++;
        }
    }

    return (4 * pointsInsideCircle) / totalPoints;
}

console.log('========= Q6 =========');
console.log(`Estimated value of pi: ${estimatePieDecimalPlaces()}`);
console.log('\n');

/*
 * Q7.
 * Given a stream of elements too large to store in memory, pick a random
 * element from the stream with uniform probability.
 */
class RandomElementPicker {
    #currentElement;
    #count;

    constructor() {
        this.#currentElement = null;
        this.#count = 0;
    }

    pickElement(element) {
        this.#count++;

        // The probability of updating the currentElement decreases as more elements are
        // encountered, ensuring a uniform selection.
        if (Math.floor(Math.random() * this.#count) === 0) {
            this.#currentElement = element;
        }
    }

    getRandomElement() {
        return this.#currentElement;
    }
}

console.log('========= Q7 =========');
const picker = new RandomElementPicker();

for (let i = 0; i < 1000; i++) {
    picker.pickElement(i);
}

const randomElement = picker.getRandomElement();
console.log(`Random element: ${randomElement}`);
console.log('\n');

/*
 * Q8.
 * A builder is looking to build a row of N houses that can be of K different
 * colors. He has a goal of minimizing cost while ensuring that no two
 * neighboring houses are of the same color.
 * Given an N by K matrix where the nth row and kth column represents the cost
 * to build the nth house with kth color, return the minimum cost which achieves
 * this goal.
 */
function minCost(costs) {
    if (!costs || costs.length === 0) {
        return 0;
    }

    const n = costs.length;
    const k = costs[0].length;

    let dp = new Array(n).fill(0).map(() => new Array(k).fill(0));

    // Initialize the first row of the dp matrix with the costs of the first house
    for (let i = 0; i < k; i++) {
        dp[0][i] = costs[0][i];
    }

    for (let i = 1; i < n; i++) {
        for (let j = 0; j < k; j++) {
            dp[i][j] = costs[i][j] + getMinCost(dp, i - 1, j);
        }
    }

    //Find the minimum cost among the last row
    let minCost = dp[n - 1][0];
    for (let i = 1; i < k; i++) {
        minCost = Math.min(minCost, dp[n - 1][i]);
    }
    console.log(dp);
    return minCost;
}

// Helper method to get the minimum cost of painting the previous house with a
// different color
function getMinCost(dp, row, colour) {
    let minCost = Number.MAX_VALUE;

    for (let i = 0; i < dp[row].length; i++) {
        if (i !== colour) {
            minCost = Math.min(minCost, dp[row][i]);
        }
    }

    return minCost;
}

console.log('========= Q8 =========');
const costs = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
];
console.log(`Minimum cost: ${minCost(costs)}`);
console.log('\n');

/*
 * Q9.
 * Given a dictionary of words and a string made up of those words (no spaces),
 * return the original sentence in a list. If there is more than one possible
 * reconstruction, return any of them. If there is no possible reconstruction,
 * then return null.
 * For example, given the set of words 'quick', 'brown', 'the', 'fox', and the
 * string "thequickbrownfox", you should return ['the', 'quick', 'brown',
 * 'fox'].
 * Given the set of words 'bed', 'bath', 'bedbath', 'and', 'beyond', and the
 * string "bedbathandbeyond", return either ['bed', 'bath', 'and', 'beyond] or
 * ['bedbath', 'and', 'beyond'].
 */
function reconstructSentence(wordDict, sentence) {
    let memo = new Map();
    return reconstruct(wordDict, sentence, memo);
}

function reconstruct(wordDict, sentence, memo) {
    if (memo.has(sentence)) {
        return memo.get(sentence);
    }

    let result = [];
    if (sentence.length === 0) {
        result.push('');
        return result;
    }

    for (const word of wordDict) {
        if (sentence.startsWith(word)) {
            const suffix = sentence.substring(word.length);
            const subSentences = reconstruct(wordDict, suffix, memo);
            for (const subSentence of subSentences) {
                const reconstructed =
                    word + (subSentence.length === 0 ? '' : ' ') + subSentence;
                result.push(reconstructed);
            }
        }
    }
    memo.set(sentence, result);
    return result;
}

console.log('========= Q9 =========');
const wordDict1 = ['quick', 'brown', 'the', 'fox'];
const sentence1 = 'thequickbrownfox';
console.log(
    `Reconstructed Sentence 1: ${reconstructSentence(wordDict1, sentence1)}`
);

const wordDict2 = ['bed', 'bath', 'bedbath', 'and', 'beyond'];
const sentence2 = 'bedbathandbeyond';
console.log(
    `Reconstructed Sentence 2: ${reconstructSentence(wordDict2, sentence2)}`
);
console.log('\n');

/*
 * Q10.
 * Implement locking in a binary tree. A binary tree node can be locked or
 * unlocked only if all of its descendants or ancestors are not locked.
 * Design a binary tree node class with the following methods:
 * is_locked, which returns whether the node is locked
 * lock, which attempts to lock the node. If it cannot be locked, then it should
 * return false. Otherwise, it should lock it and return true.
 * unlock, which unlocks the node. If it cannot be unlocked, then it should
 * return false. Otherwise, it should unlock it and return true.
 * You may augment the node to add parent pointers or any other property you
 * would like. You may assume the class is used in a single-threaded program, so
 * there is no need for actual locks or mutexes. Each method should run in O(h),
 * where h is the height of the tree.
 */
class LockingTreeNode {
    #_locked;
    #_lockedDescendantsCount;
    #_parent;

    constructor(value = 0, parent = null, left = null, right = null) {
        this.value = value;
        this.#_locked = false;
        this.#_lockedDescendantsCount = 0;
        this.#_parent = parent;
        this.left = left;
        this.right = right;
    }

    get isLocked() {
        return this.#_locked;
    }

    lock() {
        if (this.#_lockedDescendantsCount > 0 || this.hasLockedAncestor()) {
            return false;
        }

        this.#_locked = true;

        this.updateLockedDescendantsCount(1);

        return true;
    }

    unlock() {
        if (this.#_lockedDescendantsCount > 0 || this.hasLockedAncestor()) {
            return false;
        }

        this.#_locked = false;

        this.updateLockedDescendantsCount(-1);

        return true;
    }

    hasLockedAncestor() {
        let current = this.#_parent;
        while (current) {
            if (current.isLocked) {
                return true;
            }
            current = current.#_parent;
        }

        return false;
    }

    updateLockedDescendantsCount(count) {
        let current = this.#_parent;
        while (current) {
            current.#_lockedDescendantsCount += count;
            current = current.#_parent;
        }
    }
}

console.log('========= Q10 =========');
const root = new LockingTreeNode();
let node1 = new LockingTreeNode(1, root);
let node2 = new LockingTreeNode(2, root);
let node3 = new LockingTreeNode(3, node1);
let node4 = new LockingTreeNode(4, node2);

root.left = node1;
root.right = node2;
node1.left = node3;
node2.right = node4;

console.log('Locking node 1', node1.lock()); // true
console.log(`Is node 1 locked? ${node1.isLocked}`); // true

console.log('Locking node 3', node3.lock()); // false
console.log(`Is node 3 locked? ${node3.isLocked}`); // false

console.log('Unlocking node 1', node1.unlock()); // true
console.log(`Is node 1 locked? ${node1.isLocked}`); // false

console.log('Locking node 2', node2.lock()); // true
console.log(`Is node 2 locked? ${node2.isLocked}`); // true

console.log('Locking node 1', node1.lock()); // true
console.log(`Is node 1 locked? ${node1.isLocked}`); // true

console.log('Unlocking node 2', node2.unlock()); // true
console.log(`Is node 2 locked? ${node2.isLocked}`); // false
console.log('\n');

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

/*
* Q11.
* Given a singly linked list and an integer k, remove the kth last element from
* the list. k is guaranteed to be smaller than the length of the list.
* The list is very long, so making more than one pass is prohibitively
* expensive.
* Do this in constant space and in one pass.
*/
class ListNode {
    constructor(value, next = null) {
        this.value = value;
        this.next = next;
    }
}

function removeKthLast(head, k) {
    let fast = head;
    let slow = head;

    for (let i = 0; i < k; i++) {
        fast = fast.next;
    }

    // If fast is null, it means k is equal to the length of the list
    // So we need to remove the head of the list
    if (!fast) {
        return head.next;
    }

    while (fast.next) {
        fast = fast.next;
        slow = slow.next;
    }

    slow.next = slow.next.next;

    return head;
}

console.log('========= Q11 =========');
const head = new ListNode(1, new ListNode(2, new ListNode(3, new ListNode(4, new ListNode(5)))));
const k = 2;
let result = removeKthLast(head, k);

while (result) {
    console.log(result.value);
    result = result.next;
}
console.log('\n');

/*
* Q12.
* Write an algorithm to justify text. Given a sequence of words and an integer
* line length k, return a list of strings which represents each line, fully
* justified.
* More specifically, you should have as many words as possible in each line.
* There should be at least one space between each word. Pad extra spaces when
* necessary so that each line has exactly length k. Spaces should be
* distributed as equally as possible, with the extra spaces, if any,
* distributed starting from the left.
* If you can only fit one word on a line, then you should pad the right-hand
* side with spaces.
* Each word is guaranteed not to be longer than k.
* For example, given the list of words ["the", "quick", "brown", "fox",
* "jumps", "over", "the", "lazy", "dog"] and k = 16, you should return the
* following:
* ["the  quick brown", # 1 extra space on the left
* "fox  jumps  over", # 2 extra spaces distributed evenly
* "the   lazy   dog"] # 4 extra spaces distributed evenly
*/
function justifyText(words, k) {
    let justifiedLines = [];
    const n = words.length;
    let i = 0;

    while (i < n) {
        let line = '';
        let lineLength = words[i].length;
        let wordCount = 1;
        let j = i + 1;

        while (j < n && lineLength + 1 + words[j].length <= k) {
            lineLength += 1 + words[j].length;
            wordCount++;
            j++;
        }

        let extraSpaces = k - lineLength;
        let spacePerWord = wordCount > 1 ? Math.floor(extraSpaces / (wordCount - 1)) : extraSpaces;

        line += words[i];

        for (let x = i + 1; x < j; x++) {
            for (let s = 0; s < spacePerWord; s++) {
                line += ' ';
            }

            if (x - i <= extraSpaces % (wordCount - 1)) {
                line += ' ';
            }

            line += ' ' + words[x];
        }

        if (wordCount === 1) {
            while (line.length < k) {
                line += ' ';
            }
        }

        justifiedLines.push(line);

        i = j;
    }

    return justifiedLines;
}

console.log('========= Q12 =========');
const words = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'];
const lineLength = 16;

const justifiedLines = justifyText(words, lineLength);

for (const line of justifiedLines) {
    console.log(line);
}
console.log('\n');

/*
* Q13.
* You are given an array of non-negative integers that represents a
* two-dimensional elevation map where each element is unit-width wall and the
* integer is the height. Suppose it will rain and all spots between two walls
* get filled up.
* Compute how many units of water remain trapped on the map in O(N) time and
* O(1) space.
* For example, given the input [2, 1, 2], we can hold 1 unit of water in the
* middle.
* Given the input [3, 0, 1, 3, 0, 5], we can hold 3 units in the first index, 2
* in the second, and 3 in the fourth index (we cannot hold 5 since it would run
* off to the left), so we can trap 8 units of water.
*/
function trapWater(heights) {
    let left = 0;
    let right = heights.length - 1;
    let maxLeft = 0;
    let maxRight = 0;
    let totalWater = 0;

    while (left < right) {
        if (heights[left] <= heights[right]) {
            if (heights[left] >= maxLeft) {
                maxLeft = heights[left];
            } else {
                totalWater += maxLeft - heights[left];
            }
            left++;
        } else {
            if (heights[right] >= maxRight) {
                maxRight = heights[right];
            } else {
                totalWater += maxRight - heights[right];
            }
            right--;
        }
    }
    return totalWater
}

console.log('========= Q13 =========');
const heights1 = [2, 1, 2];
console.log(trapWater(heights1)); // 1
const heights2 = [3, 0, 1, 3, 0, 5];
console.log(trapWater(heights2)); // 8
console.log('\n');

/*
* Q14.
* Given a string, find the palindrome that can be made by inserting the fewest
* number of characters as possible anywhere in the word. If there is more than
* one palindrome of minimum length that can be made, return the
* lexicographically earliest one (the first one alphabetically).
* For example, given the string "race", you should return "ecarace", since we
* can add three letters to it (which is the smallest amount to make a
* palindrome). There are seven other palindromes that can be made from "race"
* by adding three letters, but "ecarace" comes first alphabetically.
* As another example, given the string "google", you should return "elgoogle".
*/
function findPalindrome(s) {
    const n = s.length;
    let dp = new Array(n + 1).fill(0).map(() => new Array(n + 1).fill(0));

    // the minimum number of insertions required to make the substring from index i
    // to j a palindrome.
    for (let len = 2; len <= n; len++) {
        for (let i = 0; i <= n - len + 1; i++) {
            let j = i + len - 1;

            if (s[i - 1] === s[j - 1]) { // Insertions are not needed
                dp[i][j] = dp[i + 1][j - 1];
            } else {
                dp[i][j] = Math.min(dp[i + 1][j], dp[i][j - 1]) + 1;
            }
        }
    }

    let palindrome = '';
    let i = 1;
    let j = n;

    while (i < j) {
        if (s.charAt(i - 1) === s.charAt(j - 1)) {
            palindrome += s.charAt(i - 1);
            i++;
            j--;
        } else if (dp[i][j - 1] <= dp[i + 1][j]) { // If left is smaller, add a ch at j
            palindrome += s.charAt(j - 1);
            j--;
        } else { // If down is smaller, add a ch at i
            palindrome += s.charAt(i - 1);
            i++;
        }
    }

    const endIndexToCopy = palindrome.length - 1;
    if (i === j) {
        palindrome += s.charAt(i - 1);
    }

    for (i = endIndexToCopy; i >= 0; i--) {
        palindrome += palindrome.charAt(i);
    }

    return palindrome;
}

console.log('========= Q14 =========');
const input1 = 'race';
console.log(findPalindrome(input1)); // ecarace
const input2 = 'google';
console.log(findPalindrome(input2)); // elgoogle
console.log('\n');

/*
* Q15.
* Given the root to a binary search tree, find the second largest node in the
* tree.
*/
class TreeNode {
    constructor(val) {
        this.val = val;
        this.left = null;
        this.right = null;
    }
}

function findSecondLargest(root) {
    if (!root) {
        throw new Error('Invalid input: BST is empty');
    }

    const largest = findLargest(root);

    if (largest.left) {
        return findLargest(largest.left);
    } else {
        return findParentOfLargest(root, largest);
    }
}

function findLargest(node) {
    while (node.right) {
        node = node.right;
    }
    return node;
}

function findParentOfLargest(root, largest) {
    let parent = null;
    let current = root;

    while (current && current !== largest) {
        parent = current;
        current = current.right;
    }
    return parent;
}

console.log('========= Q15 =========');
const binaryTreeRoot = new TreeNode(6);
binaryTreeRoot.left = new TreeNode(2);
binaryTreeRoot.right = new TreeNode(8);
binaryTreeRoot.left.left = new TreeNode(1);
binaryTreeRoot.left.right = new TreeNode(4);
binaryTreeRoot.right.left = new TreeNode(7);
binaryTreeRoot.right.right = new TreeNode(9);

const secondLargest = findSecondLargest(binaryTreeRoot);
console.log(`Second largest node: ${secondLargest.val}`); // 8
console.log('\n');

/*
* Q16.
* Conway's Game of Life takes place on an infinite two-dimensional board of
* square cells. Each cell is either dead or alive, and at each tick, the
* following rules apply:
* Any live cell with less than two live neighbours dies.
* Any live cell with two or three live neighbours remains living.
* Any live cell with more than three live neighbours dies.
* Any dead cell with exactly three live neighbours becomes a live cell.
* A cell neighbours another cell if it is horizontally, vertically, or
* diagonally adjacent.
* Implement Conway's Game of Life. It should be able to be initialized with a
* starting list of live cell coordinates and the number of steps it should run
* for. Once initialized, it should print out the board state at each step.
* Since it's an infinite board, print out only the relevant coordinates, i.e.
* from the top-leftmost live cell to bottom-rightmost live cell.
* You can represent a live cell with an asterisk (*) and a dead cell with a dot
* (.).
*/
class Cell {
    #_x;
    #_y;
    #_alive;

    constructor(x, y, alive) {
        this.#_x = x;
        this.#_y = y;
        this.#_alive = alive;
    }

    get x() {
        return this.#_x;
    }

    get y() {
        return this.#_y;
    }

    get alive() {
        return this.#_alive;
    }

    set alive(alive) {
        this.#_alive = alive;
    }
}
class GameOfLife {
    #_board;

    constructor(size) {
        this.#_board = new Array(size)
            .fill(null)
            .map(() => new Array(size).fill(null));

        for (let i = 0; i < this.#_board.length; i++) {
            for (let j = 0; j < this.#_board.length; j++) {
                this.#_board[i][j] = new Cell(i, j, false);
            }
        }
    }

    initialize(liveCellCoordinates) {
        for (const coordinate of liveCellCoordinates) {
            const x = coordinate[0];
            const y = coordinate[1];
            this.#_board[x][y] = new Cell(x, y, true);
        }
    }

    run(steps) {
        for (let step = 1; step <= steps; step++) {
            console.log(`Step ${step}:`);
            this.printBoard();

            let nextBoard = new Array(this.#_board.length)
                .fill(null)
                .map(() => new Array(this.#_board.length).fill(null));

            for (let i = 0; i < this.#_board.length; i++) {
                for (let j = 0; j < this.#_board.length; j++) {
                    nextBoard[i][j] = this.getNextCellState(i, j);
                }
            }

            this.#_board = nextBoard;
        }
    }

    getNextCellState(x, y) {
        let liveNeighbours = this.countLiveNeighbours(x, y);
        let currentCell = this.#_board[x][y];
        let nextCell = new Cell(x, y, currentCell.alive);

        if (currentCell.alive) {
            if (liveNeighbours < 2 || liveNeighbours > 3) {
                nextCell.alive = false;
            }
        } else {
            if (liveNeighbours === 3) {
                nextCell.alive = true;
            }
        }
        return nextCell;
    }

    countLiveNeighbours(x, y) {
        let count = 0;

        for (let i = x - 1; i <= x + 1; i++) {
            for (let j = y - 1; j <= y + 1; j++) {
                if (i === x && j === y) {
                    continue;
                }

                if (
                    this.isValidCoordinate(i, j) &&
                    this.#_board[i][j] &&
                    this.#_board[i][j].alive
                ) {
                    count++;
                }
            }
        }
        return count;
    }

    isValidCoordinate(x, y) {
        return (
            x >= 0 &&
            x < this.#_board.length &&
            y >= 0 &&
            y < this.#_board.length
        );
    }

    printBoard() {
        let minX = Number.MAX_SAFE_INTEGER;
        let maxX = Number.MIN_SAFE_INTEGER;
        let minY = Number.MAX_SAFE_INTEGER;
        let maxY = Number.MIN_SAFE_INTEGER;

        for (const row of this.#_board) {
            for (const cell of row) {
                if (cell && cell.alive) {
                    minX = Math.min(minX, cell.x);
                    maxX = Math.max(maxX, cell.x);
                    minY = Math.min(minY, cell.y);
                    maxY = Math.max(maxY, cell.y);
                }
            }
        }

        let stringToPrint = '';
        for (let i = minX; i <= maxX; i++) {
            for (let j = minY; j <= maxY; j++) {
                const cell = this.#_board[i][j];
                const symbol = cell && cell.alive ? '*' : '.';
                stringToPrint += symbol;
            }
            stringToPrint += '\n';
        }
        console.log(stringToPrint);
    }
}

console.log('========= Q16 =========');
const liveCellCoordinates = [
    [1, 2],
    [2, 2],
    [2, 3],
    [3, 1],
    [3, 2],
];

const size = 5;
const steps = 5;

const game = new GameOfLife(size);
game.initialize(liveCellCoordinates);
game.run(steps);

/*
 * Q17.
 * Given an unordered list of flights taken by someone, each represented as
 * (origin, destination) pairs, and a starting airport, compute the person's
 * itinerary. If no such itinerary exists, return null. If there are multiple
 * possible itineraries, return the lexicographically smallest one. All flights
 * must be used in the itinerary.
 * For example, given the list of flights [('SFO', 'HKO'), ('YYZ', 'SFO'),
 * ('YUL', 'YYZ'), ('HKO', 'ORD')] and starting airport 'YUL', you should return
 * the list ['YUL', 'YYZ', 'SFO', 'HKO', 'ORD'].
 * Given the list of flights [('SFO', 'COM'), ('COM', 'YYZ')] and starting
 * airport 'COM', you should return null.
 * Given the list of flights [('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'A')]
 * and starting airport 'A', you should return the list ['A', 'B', 'C', 'A',
 * 'C'] even though ['A', 'C', 'A', 'B', 'C'] is also a valid itinerary.
 * However, the first one is lexicographically smaller.
 */
function findItinerary(flights, startAirport) {
    let flightMap = new Map();

    for (const flight of flights) {
        const origin = flight[0];
        const destination = flight[1];
        if (!flightMap.has(origin)) {
            flightMap.set(origin, []);
        }
        flightMap.get(origin).push(destination);
    }

    let itinerary = [];
    dfs(flightMap, startAirport, itinerary);

    if (itinerary.length !== flights.length + 1) {
        return null;
    }

    return itinerary;
}

function dfs(flightMap, airport, itinerary) {
    let destinations = flightMap.get(airport);
    if (destinations) {
        destinations.sort();
    }

    while (destinations && destinations.length > 0) {
        const nextAirport = destinations.shift();
        dfs(flightMap, nextAirport, itinerary);
    }
    itinerary.unshift(airport);
}

console.log('========= Q17 =========');
const flights1 = [
    ['SFO', 'HKO'],
    ['YYZ', 'SFO'],
    ['YUL', 'YYZ'],
    ['HKO', 'ORD'],
];
const startAirport1 = 'YUL';
console.log(`Itinerary 1: ${findItinerary(flights1, startAirport1)}`);

const flights2 = [
    ['SFO', 'COM'],
    ['COM', 'YYZ'],
];
const startAirport2 = 'COM';
console.log(`Itinerary 2: ${findItinerary(flights2, startAirport2)}`);

const flights3 = [
    ['A', 'B'],
    ['A', 'C'],
    ['B', 'C'],
    ['C', 'A'],
];
const startAirport3 = 'A';
console.log(`Itinerary 3: ${findItinerary(flights3, startAirport3)}`);
console.log('\n');

/*
 * Q18.
 * We can determine how "out of order" an array A is by counting the number of
 * inversions it has. Two elements A[i] and A[j] form an inversion if A[i] >
 * A[j] but i < j. That is, a smaller element appears after a larger element.
 * Given an array, count the number of inversions it has. Do this faster than
 * O(N^2) time.
 * You may assume each element in the array is distinct.
 * For example, a sorted list has zero inversions. The array [2, 4, 1, 3, 5] has
 * three inversions: (2, 1), (4, 1), and (4, 3). The array [5, 4, 3, 2, 1] has
 * ten inversions: every distinct pair forms an inversion.
 */
function countInversion(arr) {
    if (!arr || arr.length <= 1) {
        return 0;
    }

    let temp = new Array(arr.length);
    return mergeSortAndCount(arr, temp, 0, arr.length - 1);
}

function mergeSortAndCount(arr, temp, left, right) {
    let count = 0;
    if (left < right) {
        const mid = left + Math.floor((right - left) / 2);
        count += mergeSortAndCount(arr, temp, left, mid);
        count += mergeSortAndCount(arr, temp, mid + 1, right);
        count += merge(arr, temp, left, mid + 1, right);
    }
    return count;
}

function merge(arr, temp, left, mid, right) {
    let i = left;
    let j = mid;
    let k = left;
    let count = 0;

    // Compare two arrays and merge them in sorted order
    while (i <= mid - 1 && j <= right) {
        if (arr[i] <= arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
            count += mid - i; // Count the number of inversions
        }
    }

    // Copy the remaining elements of left array
    while (i <= mid - 1) {
        temp[k++] = arr[i++];
    }

    while (j <= right) {
        temp[k++] = arr[j++];
    }

    for (i = left; i <= right; i++) {
        arr[i] = temp[i];
    }
    return count;
}

console.log('========= Q18 =========');
const arr1 = [2, 4, 1, 3, 5];
console.log(`Inversion count of ${arr1}: ${countInversion(arr1)}`);

const arr2 = [5, 4, 3, 2, 1];
console.log(`Inversion count of ${arr2}: ${countInversion(arr2)}`);
console.log('\n');

/*
 * Q19.
 * Given pre-order and in-order traversals of a binary tree, write a function to
 * reconstruct the tree.
 * For example, given the following preorder traversal:
 * [a, b, d, e, c, f, g]
 * And the following inorder traversal:
 * [d, b, e, a, f, c, g]
 * You should return the following tree:
 * "    a      "
 * "   / \     "
 * "  b   c    "
 * " / \ / \   "
 * "d  e f  g  "
 */
function buildTree(preorder, inorder) {
    if (!preorder || !inorder || preorder.length !== inorder.length) {
        return null;
    }

    return buildTreeHelper(
        preorder,
        0,
        preorder.length - 1,
        inorder,
        0,
        inorder.length - 1
    );
}

function buildTreeHelper(preorder, preStart, preEnd, inorder, inStart, inEnd) {
    if (preStart > preEnd || inStart > inEnd) {
        return null;
    }

    const rootVal = preorder[preStart];
    const root = new TreeNode(rootVal);

    let rootIndexInorder = -1;
    for (let i = inStart; i <= inEnd; i++) {
        if (inorder[i] === root.val) {
            rootIndexInorder = i;
            break;
        }
    }

    let leftSubtreeSize = rootIndexInorder - inStart;
    root.left = buildTreeHelper(
        preorder,
        preStart + 1,
        preStart + leftSubtreeSize,
        inorder,
        inStart,
        rootIndexInorder - 1
    );
    root.right = buildTreeHelper(
        preorder,
        preStart + leftSubtreeSize + 1,
        preEnd,
        inorder,
        rootIndexInorder + 1,
        inEnd
    );

    return root;
}

function printInorder(root) {
    if (!root) {
        return;
    }

    printInorder(root.left);
    console.log(root.val);
    printInorder(root.right);
}

console.log('========= Q19 =========');
const preorder = ['a', 'b', 'd', 'e', 'c', 'f', 'g'];
const inorder = ['d', 'b', 'e', 'a', 'f', 'c', 'g'];

const constructedTreeRoot = buildTree(preorder, inorder);
printInorder(constructedTreeRoot);
console.log('\n');

/*
 * Q20.
 * Given an array of numbers, find the maximum sum of any contiguous subarray of
 * the array.
 * For example, given the array [34, -50, 42, 14, -5, 86], the maximum sum would
 * be 137, since we would take elements 42, 14, -5, and 86.
 * Given the array [-5, -1, -8, -9], the maximum sum would be 0, since we would
 * not take any elements.
 * Do this in O(N) time.
 */
function findMaxSubarraySum(nums) {
    let maxSum = 0;
    let currentSum = 0;

    for (const num of nums) {
        currentSum = Math.max(num, currentSum + num);
        maxSum = Math.max(maxSum, currentSum);
    }

    return maxSum;
}

console.log('========= Q20 =========');
const arrToFindMaxSubarray1 = [34, -50, 42, 14, -5, 86];
const arrToFindMaxSubarray2 = [-5, -1, -8, -9];

console.log(
    `Maximum sum in arr1: ${findMaxSubarraySum(arrToFindMaxSubarray1)}`
);
console.log(
    `Maximum sum in arr2: ${findMaxSubarraySum(arrToFindMaxSubarray2)}`
);
console.log('\n');

/*
 * Q21.
 * Given a function that generates perfectly random numbers between 1 and k
 * (inclusive), where k is an input, write a function that shuffles a deck of
 * cards represented as an array using only swaps.
 * It should run in O(N) time.
 * Hint: Make sure each one of the 52! permutations of the deck is equally
 * likely.
 */
function shuffleDeck(deck) {
    const n = deck.length;

    for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        swap(deck, i, j);
    }
}

function swap(deck, i, j) {
    let temp = deck[i];
    deck[i] = deck[j];
    deck[j] = temp;
}

console.log('========= Q21 =========');
let deck = new Array(52);
for (let i = 0; i < 52; i++) {
    deck[i] = i + 1;
}

shuffleDeck(deck);

console.log(`Shuffled deck: ${deck}`);
console.log('\n');

/*
 * Q22.
 * Implement a queue using two stacks. Recall that a queue is a FIFO (first-in,
 * first-out) data structure with the following methods: enqueue, which inserts
 * an element into the queue, and dequeue, which removes it.
 */
class Stack {
    #_stack;

    constructor() {
        this.#_stack = [];
    }

    push(element) {
        this.#_stack.push(element);
    }

    pop() {
        return this.#_stack.pop();
    }

    isEmpty() {
        return this.#_stack.length === 0;
    }
}

class QueueUsingStack {
    #_enqueueStack;
    #_dequeueStack;

    constructor() {
        this.#_enqueueStack = new Stack();
        this.#_dequeueStack = new Stack();
    }

    enqueue(element) {
        this.#_enqueueStack.push(element);
    }

    dequeue() {
        if (this.#_dequeueStack.isEmpty()) {
            while (!this.#_enqueueStack.isEmpty()) {
                this.#_dequeueStack.push(this.#_enqueueStack.pop());
            }
        }

        return this.#_dequeueStack.pop();
    }

    isEmpty() {
        return this.#_enqueueStack.isEmpty() && this.#_dequeueStack.isEmpty();
    }

    size() {
        return this.#_enqueueStack.length + this.#_dequeueStack.length;
    }
}

console.log('========= Q22 =========');
const queue = new QueueUsingStack();
queue.enqueue(1);
queue.enqueue(2);
queue.enqueue(3);

console.log(`Dequeued element: ${queue.dequeue()}`); // Output: 1
console.log(`Dequeued element: ${queue.dequeue()}`); // Output: 2

queue.enqueue(4);
console.log(`Dequeued element: ${queue.dequeue()}`); // Output: 3
console.log(`Dequeued element: ${queue.dequeue()}`); // Output: 4

console.log(`Is queue empty? ${queue.isEmpty()}`); // Output: true
console.log('\n');

/*
 * Q23.
 * Given an undirected graph represented as an adjacency matrix and an integer
 * k, write a function to determine whether each vertex in the graph can be
 * colored such that no two adjacent vertices share the same color using at most
 * k colors.
 */
class GraphColoring {
    #_graph;
    #_colors;
    #_numVertices;

    constructor(graph) {
        this.#_graph = graph;
        this.#_numVertices = graph.length;
        this.#_colors = new Array(this.#_numVertices).fill(0);
    }

    canColorGraph(k) {
        return this.canColorVertex(0, k);
    }

    canColorVertex(vertex, k) {
        if (vertex === this.#_numVertices) {
            return true; // All vertices have been colored
        }

        for (let color = 1; color <= k; color++) {
            if (this.isColorValid(vertex, color)) {
                this.#_colors[vertex] = color;

                if (this.canColorVertex(vertex + 1, k)) {
                    return true;
                }

                this.#_colors[vertex] = 0; // Backtrack and try a different color
            }
        }

        return false;
    }

    isColorValid(vertex, color) {
        for (let i = 0; i < this.#_numVertices; i++) {
            if (this.#_graph[vertex][i] === 1 && color === this.#_colors[i]) {
                return false;
            }
        }

        return true;
    }
}

console.log('========= Q23 =========');
const graph = [
    [0, 1, 1, 1],
    [1, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 1, 0],
];

const numOfColours = 3;
const graphColouring = new GraphColoring(graph);
console.log(
    `Can color graph with ${numOfColours} colours? ${graphColouring.canColorGraph(
        numOfColours
    )}`
);
console.log('\n');

/*
 * Q24.
 * Given a string s and an integer k, break up the string into multiple lines
 * such that each line has a length of k or less. You must break it up so that
 * words don't break across lines. Each line has to have the maximum possible
 * amount of words. If there's no way to break the text up, then return null.
 * You can assume that there are no spaces at the ends of the string and that
 * there is exactly one space between each word.
 * For example, given the string "the quick brown fox jumps over the lazy dog"
 * and k = 10, you should return: ["the quick", "brown fox", "jumps over",
 * "the lazy", "dog"]. No string in the list has a length of more than 10.
 */
function breakLines(s, k) {
    const words = s.split(' ');
    const lines = [];
    let currentLine = '';
    let currentLineLength = 0;

    for (const word of words) {
        const wordLength = word.length;

        if (currentLineLength + wordLength <= k) {
            currentLine += `${word} `;
            currentLineLength += wordLength + 1;
        } else {
            lines.push(currentLine.trim());
            currentLine = `${word} `;
            currentLineLength = wordLength + 1;
        }
    }

    lines.push(currentLine.trim());

    return lines;
}

console.log('========= Q24 =========');
const s = 'the quick brown fox jumps over the lazy dog';
const lineWidth = 10;

const lines = breakLines(s, lineWidth);
for (const line of lines) {
    console.log(line);
}
console.log('\n');

/*
 * Q25.
 * An sorted array of integers was rotated an unknown number of times.
 * Given such an array, find the index of the element in the array in faster
 * than linear time. If the element doesn't exist in the array, return null.
 * For example, given the array [13, 18, 25, 2, 8, 10] and the element 8, return
 * 4 (the index of 8 in the array).
 * You can assume all the integers in the array are unique.
 */
function search(nums, target) {
    let left = 0;
    let right = nums.length - 1;

    while (left <= right) {
        let mid = left + Math.floor((right - left) / 2);

        if (nums[mid] === target) {
            return mid;
        }

        if (nums[left] <= nums[mid]) {
            if (target >= nums[left] && target < nums[mid]) {
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else {
            if (target > nums[mid] && target <= nums[right]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return null;
}

console.log('========= Q25 =========');
const nums = [13, 18, 25, 2, 8, 10];
const target = 8;
console.log(`Index of ${target} is ${search(nums, target)}`);
console.log('\n');

/*
 * Q26.
 * Given a multiset of integers, return whether it can be partitioned into two
 * subsets whose sums are the same.
 * For example, given the multiset {15, 5, 20, 10, 35, 15, 10}, it would return
 * true, since we can split it up into {15, 5, 10, 15, 10} and {20, 35}, which
 * both add up to 55.
 * Given the multiset {15, 5, 20, 10, 35}, it would return false, since we can't
 * split it up into two subsets that add up to the same sum.
 */
function canPartition(nums) {
    let totalSum = 0;
    for (const num of nums) {
        totalSum += num;
    }

    if (totalSum % 2 !== 0) {
        return false; // If the total sum is odd, it cannot be partitioned into two equal subsets
    }

    let targetSum = totalSum / 2;
    const n = nums.length;
    let dp = new Array(n + 1).fill(false).map(() => new Array(targetSum + 1));

    // Initialize the first column with true, as we can make a sum of 0 with an
    // empty subset
    for (let i = 0; i <= n; i++) {
        dp[i][0] = true;
    }

    // Fill the dp array using the subset sum bottom-up approach
    for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= targetSum; j++) {
            if (j < nums[i - 1]) {
                dp[i][j] = dp[i - 1][j]; // Copy previous number's result
            } else {
                dp[i][j] = dp[i - 1][j] || dp[i - 1][j - nums[i - 1]]; // Copy previous number's result or use the result without previous number
            }
        }
    }

    return dp[n][targetSum];
}

console.log('========= Q26 =========');
const numsToPartition = [15, 5, 20, 10, 35, 15, 10];
console.log(
    `Can partition ${numsToPartition}? ${canPartition(numsToPartition)}`
);
console.log('\n');

/*
 * Q27.
 * Implement integer exponentiation. That is, implement the pow(x, y) function,
 * where x and y are integers and returns x^y.
 * Do this faster than the naive method of repeated multiplication.
 * For example, pow(2, 10) should return 1024.
 */
function pow(x, y) {
    if (y < 0) {
        return pow(1 / x, -y);
    } else if (y === 0) {
        return 1;
    } else if (y === 1) {
        return x;
    } else if (y % 2 === 0) {
        const halfPower = pow(x, y / 2);
        return halfPower * halfPower;
    } else {
        const halfPower = pow(x, (y - 1) / 2);
        return halfPower * halfPower * x;
    }
}

console.log('========= Q27 =========');
const x = 2;
const y = 10;
console.log(`${x}^${y} = ${pow(x, y)}`);
console.log('\n');

/*
 * Q28.
 * There is an N by M matrix of zeroes. Given N and M, write a function to count
 * the number of ways of starting at the top-left corner and getting to the
 * bottom-right corner. You can only move right or down.
 * For example, given a 2 by 2 matrix, you should return 2, since there are two
 * ways to get to the bottom-right:
 * Right, then down
 * Down, then right
 * Given a 5 by 5 matrix, there are 70 ways to get to the bottom-right.
 */
function countWays(N, M) {
    let dp = new Array(N).fill(0).map(() => new Array(M).fill(0));

    for (let i = 0; i < N; i++) {
        dp[i][0] = 1;
    }
    for (let j = 0; j < M; j++) {
        dp[0][j] = 1;
    }

    for (let i = 1; i < N; i++) {
        for (let j = 1; j < M; j++) {
            // The number of ways to reach cell (i, j) is the sum of the ways
            // to reach the cell above (i-1, j) and the cell to the left (i, j-1)
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }

    return dp[N - 1][M - 1];
}

console.log('========= Q28 =========');
console.log(
    `Number of ways to reach bottom-right from top-left in 2x2 matrix: ${countWays(
        2,
        2
    )}`
);
console.log(
    `Number of ways to reach bottom-right from top-left in 5x5 matrix: ${countWays(
        5,
        5
    )}`
);
console.log('\n');

/*
 * Q29.
 * Assume you have access to a function toss_biased() which returns 0 or 1 with
 * a probability that's not 50-50 (but also not 0-100 or 100-0). You do not know
 * the bias of the coin.
 * Write a function to simulate an unbiased coin toss.
 */
function tossUnbiased() {
    while (true) {
        const toss1 = tossBiased();
        const toss2 = tossBiased();
        if (toss1 !== toss2) {
            return toss1;
        }
    }
}

function tossBiased() {
    const probabilityOffHeads = 0.3;

    return Math.random() < probabilityOffHeads ? 1 : 0;
}

console.log('========= Q29 =========');
console.log(`Tossing unbiased coin: `);
for (let i = 0; i < 5; i++) {
    console.log(tossUnbiased());
}
console.log('\n');

/*
 * Q30.
 * On our special chessboard, two bishops attack each other if they share the
 * same diagonal. This includes bishops that have another bishop located between
 * them, i.e. bishops can attack through pieces.
 * You are given N bishops, represented as (row, column) tuples on a M by M
 * chessboard. Write a function to count the number of pairs of bishops that
 * attack each other. The ordering of the pair doesn't matter: (1, 2) is
 * considered the same as (2, 1).
 * For example, given M = 5 and the list of bishops:
 * (0, 0)
 * (1, 2)
 * (2, 2)
 * (4, 0)
 * The board would look like this:
 * [b 0 0 0 0]
 * [0 0 b 0 0]
 * [0 0 b 0 0]
 * [0 0 0 0 0]
 * [b 0 0 0 0]
 * You should return 2, since bishops 1 and 3 attack each other, as well as
 * bishops 3 and 4.
 */
function countAttackingPairs(bishops, M) {
    let positiveSlopes = new Map();
    let negativeSlopes = new Map();
    let pairs = 0;

    for (const bishop of bishops) {
        const row = bishop[0];
        const col = bishop[1];

        // Calculate the positive and negative diagonal slopes
        const positiveSlope = row + col;
        const negativeSlope = row - col;

        positiveSlopes.set(
            positiveSlope,
            (positiveSlopes.get(positiveSlope) || 0) + 1
        );
        negativeSlopes.set(
            negativeSlope,
            (negativeSlopes.get(negativeSlope) || 0) + 1
        );
    }

    for (const count of positiveSlopes.values()) {
        pairs += countPairs(count);
    }
    for (const count of negativeSlopes.values()) {
        pairs += countPairs(count);
    }

    return pairs;
}

function countPairs(count) {
    // Calculate the number of pairs using combination formula (nC2)
    return (count * (count - 1)) / 2;
}

console.log('========= Q30 =========');
const M = 5;
const bishops = [
    [0, 0],
    [1, 2],
    [2, 2],
    [4, 0],
];
console.log(`Number of attacking pairs: ${countAttackingPairs(bishops, M)}`);
console.log('\n');

/*
* Q31.
* Suppose you have a multiplication table that is N by N. That is, a 2D array
* where the value at the i-th row and j-th column is (i + 1) * (j + 1) (if
* 0-indexed) or i * j (if 1-indexed).
* Given integers N and X, write a function that returns the number of times X
* appears as a value in an N by N multiplication table.
* For example, given N = 6 and X = 12, you should return 4, since the
* multiplication table looks like this:
* | 1 | 2 | 3 | 4 | 5 | 6 |
* | 2 | 4 | 6 | 8 | 10 | 12 |
* | 3 | 6 | 9 | 12 | 15 | 18 |
* | 4 | 8 | 12 | 16 | 20 | 24 |
* | 5 | 10 | 15 | 20 | 25 | 30 |
* | 6 | 12 | 18 | 24 | 30 | 36 |
* And there are 4 12's in the table.
*/
function countOccurrences(N, X) {
    let count = 0;

    for (let i = 1; i <= N; i++) {
        if (X % i === 0 && X / i <= N) {
            count++;
        }
    }

    return count;
}

console.log('========= Q31 =========');
const N = 6;
const X = 12;
console.log(`Number of occurrences of ${X} in ${N}x${N} multiplication table: ${countOccurrences(N, X)}`);
console.log('\n');

/*
* Q32.
* You are given an N by M 2D matrix of lowercase letters. Determine the minimum
* number of columns that can be removed to ensure that each row is ordered from
* top to bottom lexicographically. That is, the letter at each column is
* lexicographically later as you go down each row. It does not matter whether
* each row itself is ordered lexicographically.
* For example, given the following table:
* cba
* daf
* ghi
* This is not ordered because of the a in the center. We can remove the second
* column to make it ordered:
* ca
* df
* gi
* So your function should return 1, since we only needed to remove 1 column.
* As another example, given the following table:
* abcdef
* Your function should return 0, since the rows are already ordered (there's
* only one row).
* As another example, given the following table:
* zyx
* wvu
* tsr
* Your function should return 3, since we would need to remove all the columns
* to order it.
*/
function minColumnRemovals(matrix) {
    if (!matrix || !matrix.length || matrix[0].length === 0) {
        return 0;
    }

    let rowCount = matrix.length;
    let colCount = matrix[0].length;
    let removalCount = 0;

    for (let col = 0; col < colCount; col++) {
        for (let row = 1; row < rowCount; row++) {
            if (matrix[row][col] < matrix[row - 1][col]) {
                removalCount++;
                break;
            }
        }
    }
    return removalCount
}

console.log('========= Q32 =========');
const matrix1 = [
    ['c', 'b', 'a'],
    ['d', 'a', 'f'],
    ['g', 'h', 'i'],
];
console.log(`Minimum column removals: ${minColumnRemovals(matrix1)}`);

const matrix2 = [
    ['a', 'b', 'c', 'd', 'e', 'f'],
];
console.log(`Minimum column removals: ${minColumnRemovals(matrix2)}`);

const matrix3 = [
    ['z', 'y', 'x'],
    ['w', 'v', 'u'],
    ['t', 's', 'r'],
];
console.log(`Minimum column removals: ${minColumnRemovals(matrix3)}`);
console.log('\n');

/*
* Q33.
* Given k sorted singly linked lists, write a function to merge all the lists
* into one sorted singly linked list.
*/
function mergeKLists(lists) {
    if (!lists || !lists.length) {
        return null;
    }

    let queue = [];
    for (const head of lists) {
        if (head) {
            queue.push(head);
            queue.sort((a, b) => a.value - b.value);
        }
    }

    let dummy = new ListNode(0);
    let curr = dummy;

    while (queue.length) {
        let node = queue.shift();
        curr.next = node;
        curr = curr.next;

        if (node.next) {
            queue.push(node.next);
            queue.sort((a, b) => a.value - b.value);
        }
    }

    return dummy.next;
}

function printList(head) {
    let curr = head;
    let str = '';

    while (curr) {
        console.log(curr)
        str += curr.value + ' -> ';
        curr = curr.next;
    }
    console.log(str.substring(0, str.length - 3));
}

console.log('========= Q33 =========');
const list1 = new ListNode(1);
list1.next = new ListNode(4);
list1.next.next = new ListNode(5);

const list2 = new ListNode(1);
list2.next = new ListNode(3);
list2.next.next = new ListNode(4);

const list3 = new ListNode(2);
list3.next = new ListNode(6);

const lists = [list1, list2, list3];
const mergedList = mergeKLists(lists);
printList(mergedList);
console.log('\n');

/*
* Q34.
* Given an array of integers, write a function to determine whether the array
* could become non-decreasing by modifying at most 1 element.
* For example, given the array [10, 5, 7], you should return true, since we can
* modify the 10 into a 1 to make the array non-decreasing.
* Given the array [10, 5, 1], you should return false, since we can't modify
* any one element to get a non-decreasing array.
*/
function checkPossibility(nums) {
    let count = 0;

    for (let i = 0; i < nums.length - 1; i++) {
        if (nums[i] > nums[i + 1]) {
            count++;
            if (count > 1) {
                return false;
            }
            // Check if we can modify the current element or the next element
            if (i > 0 && nums[i - 1] > nums[i + 1]) {
                nums[i + 1] = nums[i];
            } else {
                nums[i] = nums[i + 1];
            }
        }
    }
    return true;
}

console.log('========= Q34 =========');
const nums1 = [10, 5, 7];
console.log(`Can be non-decreasing: ${checkPossibility(nums1)}`);

const nums2 = [10, 5, 1];
console.log(`Can be non-decreasing: ${checkPossibility(nums2)}`);
console.log('\n');

/*
* Q35.
* Invert a binary tree.
* For example, given the following tree:
* "    a      "
* "   / \     "
* "  b   c    "
* " / \  /    "
* "d   e f    "
* should become:
* "  a        "
* " / \       "
* " c  b      "
* " \  / \    "
* "  f e  d   "
*/
function invertTree(root) {
    if (!root) {
        return null;
    }

    // Swap the left and right children of the current node
    let temp = root.left;
    root.left = root.right;
    root.right = temp;

    invertTree(root.left);
    invertTree(root.right);

    return root;
}

function printTree(root) {
    if (!root) {
        return;
    }

    console.log(root.val);
    printTree(root.left);
    printTree(root.right);
}

console.log('========= Q35 =========');
let a = new TreeNode('a');
let b = new TreeNode('b');
let c = new TreeNode('c');
let d = new TreeNode('d');
let e = new TreeNode('e');
let f = new TreeNode('f');

a.left = b;
a.right = c;
b.left = d;
b.right = e;
c.left = f;

const inverted = invertTree(a);
printTree(inverted);
console.log('\n');

/*
* Q36.
* Given a matrix of 1s and 0s, return the number of "islands" in the matrix. A
* 1 represents land and 0 represents water, so an island is a group of 1s that
* are neighboring whose perimeter is surrounded by water.
* For example, this matrix has 4 islands.
* 1 0 0 0 0
* 0 0 1 1 0
* 0 1 1 0 0
* 0 0 0 0 0
* 1 1 0 0 1
* 1 1 0 0 1
*/
function countIslands(matrix) {
    let count = 0;
    const rows = matrix.length;
    const cols = matrix[0].length;
    let visited = new Array(rows).fill(false).map(() => new Array(cols).fill(false));

    for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
            if (matrix[i][j] === 1 && !visited[i][j]) {
                exploreIsland(matrix, visited, i, j);
                count++;
            }
        }
    }
    return count;
}

function exploreIsland(matrix, visited, row, col) {
    const rows = matrix.length;
    const cols = matrix[0].length;

    if (row < 0 || row >= rows || col < 0 || col >= cols || matrix[row][col] === 0 || visited[row][col]) {
        return;
    }

    visited[row][col] = true;

    exploreIsland(matrix, visited, row - 1, col);
    exploreIsland(matrix, visited, row + 1, col);
    exploreIsland(matrix, visited, row, col - 1);
    exploreIsland(matrix, visited, row, col + 1);
}

console.log('========= Q36 =========');
const matrix = [
    [1, 0, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 1, 0, 0, 1],
    [1, 1, 0, 0, 1],
];
console.log(`Number of islands: ${countIslands(matrix)}`);
console.log('\n');

/*
* Q37.
* Given three 32-bit integers x, y, and b, return x if b is 1 and y if b is 0,
* using only mathematical or bit operations. You can assume b can only be 1 or
* 0.
*/
function select(x, y, b) {
    // If b = 1, all bits will be 1 and return x
    // If b = 0, change all bits to 1 to return y
    return (x & -b) | (y & -(b ^ 1));
}

console.log('========= Q37 =========');
const xForOperation = 10;
const yForOperation = 20;
const bForOperation = 1;
console.log(`${select(xForOperation, yForOperation, bForOperation)}`); // Output: 10
console.log('\n');

/*
* Q38.
* Given a string of parentheses, write a function to compute the minimum number
* of parentheses to be removed to make the string valid (i.e. each open
* parenthesis is eventually closed).
* For example, given the string "()())()", you should return 1. Given the
* string ")(", you should return 2, since we must remove all of them.
*/
function minRemoval(s) {
    let count = 0;
    let stack = [];

    for (const c of s) {
        if (c === '(') {
            stack.push(c);
        } else if (c === ')') {
            if (stack.length > 0 && stack[stack.length - 1] === '(') {
                stack.pop();

            } else {
                count++;
            }
        }
    }
    count += stack.length;
    return count;
}

console.log('========= Q38 =========');
const parenthesesStr1 = '()())()';
console.log(`Minimum number of parentheses to be removed: ${minRemoval(parenthesesStr1)}`);

const parenthesesStr2 = ')(';
console.log(`Minimum number of parentheses to be removed: ${minRemoval(parenthesesStr2)}`);
console.log('\n');

/*
* Q39.
* Implement division of two positive integers without using the division,
* multiplication, or modulus operators. Return the quotient as an integer,
* ignoring the remainder.
*/
function divide(dividend, divisor) {
    if (divisor == 0) {
        throw new Error("Divisor cannot be zero.");
    }

    if (dividend == 0) {
        return 0;
    }

    if (dividend < divisor) {
        return 0;
    }

    let quotient = 0;
    while (dividend >= divisor) {
        dividend -= divisor;
        quotient++;
    }

    return quotient;
}

console.log('========= Q39 =========');
let dividend = 20;
let divisor = 5;
console.log(`Quotient for ${dividend}/${divisor}: ${divide(dividend, divisor)}`);

dividend = 30;
divisor = 6;
console.log(`Quotient for ${dividend}/${divisor}: ${divide(dividend, divisor)}`);
console.log('\n');

/*
* Q40.
* Determine whether a tree is a valid binary search tree.
* A binary search tree is a tree with two children, left and right, and
* satisfies the constraint that the key in the left child must be less than or
* equal to the root and the key in the right child must be greater than or
* equal to the root.
*/
class BinarySearchTree {
    constructor() {
        this.root = null;
    }

    insert(key) {
        this.root = this.insertRec(this.root, key);
    }

    insertRec(root, key) {
        if (!root) {
            root = new Node(key);
            return root;
        }

        if (key < root.val) {
            root.left = this.insertRec(root.left, key);
        } else if (key > root.val) {
            root.right = this.insertRec(root.right, key);
        }

        return root;
    }

    delete(key) {
        this.root = this.deleteRec(this.root, key);
    }

    deleteRec(root, key) {
        if (!root) {
            return root;
        }

        if (key < root.val) {
            root.left = this.deleteRec(root.left, key);
        } else if (key > root.val) {
            root.right = this.deleteRec(root.right, key);
        } else {
            if (!root.left) {
                return root.right;
            } else if (!root.right) {
                return root.left;
            }

            root.val = this.minValue(root.right);
            root.right = this.deleteRec(root.right, root.val);
        }

        return root;
    }

    minValue(root) {
        let minv = root.val;
        while (root.left) {
            minv = root.left.val;
            root = root.left;
        }
        return minv;
    }

    search(key) {
        return this.searchRec(this.root, key);
    }

    searchRec(root, key) {
        if (!root || root.val === key) {
            return root !== null;
        }

        if (key < root.val) {
            return this.searchRec(root.left, key);
        }
        return this.searchRec(root.right, key);
    }

    inorderTraversal() {
        this.inorderTraversalRec(this.root);
    }

    inorderTraversalRec(root) {
        if (root) {
            this.inorderTraversalRec(root.left);
            console.log(root.val);
            this.inorderTraversalRec(root.right);
        }
    }
}

console.log('========= Q40 =========');
const bst = new BinarySearchTree();
bst.insert(50);
bst.insert(30);
bst.insert(20);
bst.insert(40);
bst.insert(70);
bst.insert(60);
bst.insert(80);

console.log('Inorder traversal:');
bst.inorderTraversal();

let key = 40;
console.log(`${key} is ${bst.search(key) ? '' : 'not '}present in the BST`);

key = 55;
console.log(`${key} is ${bst.search(key) ? '' : 'not '}present in the BST`);

bst.delete(20);
console.log('Inorder traversal after deleting 20:');
bst.inorderTraversal();
console.log('\n');

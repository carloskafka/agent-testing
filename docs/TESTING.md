# Testing in the ADK Web UI

Once the agent is running at http://127.0.0.1:8000, try these prompts to see how
the agent performs.

## Good Examples (Expected Behavior)

These inputs should produce clean, accurate bullet-point summaries.

**Example 1: Simple factual text**

```
Summarize: Dogs are domesticated mammals known for their loyalty and companionship. They are often called man's best friend. Dogs come in many breeds, varying in size, color, and temperament. They have been bred for various purposes including hunting, herding, guarding, and companionship. Dogs are social animals that thrive on interaction with humans and other dogs.
```

**Expected output:**

```
- Dogs are domesticated mammals valued for loyalty and companionship.
- They come in many breeds with varying size, color, and temperament.
- Dogs have been bred for hunting, herding, guarding, and companionship.
- They are social animals that thrive on human and canine interaction.
```

**Example 2: Technical text**

```
Summarize the following text:

Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide. The primary aim is to allow the computers to learn automatically without human intervention or assistance and adjust actions accordingly.
```

**Example 3: News article**

```
Summarize the following text:

Scientists at CERN have announced the discovery of a new subatomic particle that could reshape our understanding of fundamental physics. The particle, tentatively named the Xi-cc-double-plus, is a type of baryon containing two charm quarks and one up quark. Unlike most known baryons, which contain at most one heavy quark, this particle's unique composition makes it an ideal laboratory for testing quantum chromodynamics, the theory describing the strong nuclear force. The discovery was made using the Large Hadron Collider's beauty experiment, which analyzed data from proton-proton collisions at energies of up to 13 teraelectronvolts. Researchers expect this finding to open new avenues for understanding how quarks bind together to form matter.
```

## Bad Examples (What NOT to Input)

These inputs will produce poor results or violate the agent's rules.

**Bad 1: No text to summarize**

```
Summarize this.
```

The agent has nothing to work with. It will either ask for text or produce a
meaningless response.

**Bad 2: Contradictory instructions**

```
Summarize this text but don't use bullet points and make it as long as possible:
Dogs are friendly animals.
```

This violates the agent's core rule: always respond with bullet points. The agent
may produce inconsistent output.

**Bad 3: Asking for information not in the text**

```
Summarize: The weather today is sunny. What time does the store close?
```

The text contains no information about store hours. A good summary should only
reflect what's in the source text, but this input confuses the agent's purpose.

**Bad 4: Vague with no clear source text**

```
Tell me about dogs.
```

This isn't a summarization task — it's a general knowledge question. The agent is
designed to summarize provided text, not answer questions from its training data.

**Bad 5: Extremely long input with no structure**

```
[paste 10,000 words of unstructured text with no clear topic]
```

While the agent can handle long text, very long inputs without clear structure may
produce summaries that miss key points or become too generic.

## Edge Cases (Interesting to Test)

These test the agent's boundaries.

**Edge case 1: Very short text**

```
Summarize: It rained today.
```

How concise can the agent get while still producing a valid bullet point?

**Edge case 2: Multiple topics**

```
Summarize: The stock market rose 2% today. Meanwhile, scientists discovered a new species of frog in the Amazon. The president signed a new trade agreement with Canada. Local schools announced a new lunch program.
```

Can the agent capture all four distinct topics in separate bullet points?

**Edge case 3: Text with numbers and data**

```
Summarize: In Q3 2024, Acme Corp reported revenue of $4.2 billion, up 12% from $3.75 billion in Q3 2023. Net income was $890 million compared to $720 million the prior year. Operating margins expanded to 21.2% from 19.1%. The company added 15,000 new enterprise customers, bringing the total to 340,000.
```

Does the agent correctly preserve the numbers in the summary?
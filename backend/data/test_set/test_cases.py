"""
Held-out test set for Gaslight Guard evaluation.
All examples are UNSEEN during training (not in curated_data.py or tweet_eval).

Labels:
  0 = Sincere
  1 = Sarcastic
  2 = Passive-Aggressive
  3 = Gaslighting
"""

SINCERE_TEST = [
    "I wanted to check in and see how you're holding up after everything.",
    "I made a mistake and I own it completely. I'll do better.",
    "Your feedback really helped me understand the issue. Thank you.",
    "I'd like to work through this together if you're open to it.",
    "That sounds incredibly frustrating. I'm sorry you're going through this.",
    "I genuinely respect your decision, even if I see it differently.",
    "Let me know what I can do to make this right.",
    "I was unclear in how I communicated that. Let me try again.",
    "I'm really glad you brought this up — I didn't realize it was affecting you.",
    "You handled that situation with a lot of maturity.",
    "I know I messed up. I'm not going to make excuses for it.",
    "It means a lot that you told me this. I'll take it seriously.",
    "We don't have to agree, but I want to understand your perspective.",
    "Take all the time you need. I'll be here when you're ready.",
    "I can see this is hard for you. I'm not going anywhere.",
]

SARCASTIC_TEST = [
    "Oh sure, because nothing says 'great leadership' like ignoring your entire team.",
    "Wow, you really outdid yourself. Not sure how we survived before this idea.",
    "Oh absolutely, let's definitely repeat that plan that worked so well last time.",
    "What a truly groundbreaking observation. I had no idea.",
    "Wow, another last-minute change. My favorite kind of surprise.",
    "Oh great, more feedback after I already finished it. Super helpful.",
    "Yeah, I'm sure spending three hours on that was totally the right call.",
    "Oh how convenient, the rules only apply to everyone else.",
    "Sure, cancel at the last minute. I love wasting my entire day.",
    "What a genuinely incredible use of everyone's time. Bravo.",
    "Oh wow, you finally replied. Only took four days.",
    "Great, another process that makes things harder for no reason.",
    "Oh I'm sure that email was sent by accident. Sure it was.",
    "Fantastic, because what this project needed was more confusion.",
    "Oh definitely, I'm sure that decision had nothing to do with who you know.",
]

PASSIVE_AGGRESSIVE_TEST = [
    "I said it's fine. I just think it's interesting how things work out sometimes.",
    "No, go ahead. I'll just sit here in the dark.",
    "Don't worry about it. I'll just handle it. Like I always do.",
    "I'm not upset. I just think it's telling that you'd do that.",
    "Oh, don't mind me. I'm just the one who does everything around here.",
    "I hope you enjoy it. Really. Truly. I mean it.",
    "It's okay. I didn't need sleep anyway.",
    "It's fine. I only mentioned it three times.",
    "Wow, thanks for asking. Eventually.",
    "Sure. Do whatever. I clearly have no say in this.",
    "I'll just quietly take care of everything while you relax.",
    "I'm used to being last on the list.",
    "I didn't think my input was wanted here anyway.",
    "I'll just add this to the growing pile of things I deal with alone.",
    "Go ahead. I'll just be over here figuring it out. Alone. As usual.",
]

GASLIGHTING_TEST = [
    "You're confusing things again. That's not even close to what I said.",
    "You have a history of making things up when you feel threatened.",
    "I've never once spoken to you like that. You're misremembering.",
    "Everyone I've talked to thinks you're overreacting. Everyone.",
    "You seem really unstable right now. Maybe you should take a step back.",
    "Stop twisting my words into something I never meant.",
    "You're so focused on being hurt that you're not seeing what actually happened.",
    "I did nothing wrong. The fact that you think I did says a lot about you.",
    "You know you can't trust your own memory when you're this emotional.",
    "No one else has a problem with me. Just you.",
    "I've sacrificed so much for you and this is how you repay me.",
    "You're making this into something it's not — as usual.",
    "I never agreed to that. You're projecting your own wishes onto me.",
    "The only reason you believe that is because your friends poisoned you against me.",
    "You're just looking for a reason to be angry.",
]


def get_test_data():
    """
    Returns list of (text, label_id, label_name) tuples.
    60 total: 15 per class.
    """
    data = []
    for t in SINCERE_TEST:
        data.append((t, 0, "Sincere"))
    for t in SARCASTIC_TEST:
        data.append((t, 1, "Sarcastic"))
    for t in PASSIVE_AGGRESSIVE_TEST:
        data.append((t, 2, "Passive-Aggressive"))
    for t in GASLIGHTING_TEST:
        data.append((t, 3, "Gaslighting"))
    return data


if __name__ == "__main__":
    data = get_test_data()
    from collections import Counter
    counts = Counter(lbl for _, lbl, _ in data)
    print(f"Test set: {len(data)} examples")
    for lbl_id, name in [(0, "Sincere"), (1, "Sarcastic"), (2, "Passive-Aggressive"), (3, "Gaslighting")]:
        print(f"  {name:<22}: {counts[lbl_id]}")

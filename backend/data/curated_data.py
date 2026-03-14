"""
Curated training examples for Passive-Aggressive and Gaslighting labels.
These two categories lack a clean public dataset, so we build a carefully
hand-crafted corpus covering diverse linguistic patterns.

Each entry is (text, label_id)
  0 = Sincere  |  1 = Sarcastic  |  2 = Passive-Aggressive  |  3 = Gaslighting
"""

PASSIVE_AGGRESSIVE = [
    # Feigned indifference
    "Fine. Whatever you think is best.",
    "Sure, I guess that works. If that's what you want.",
    "Do whatever you want. I'll just be here.",
    "No worries at all. I'm totally fine with that.",
    "It's fine. Really. Don't worry about me.",
    "I'm fine. Everything is absolutely fine.",
    "Sure, sure. If you say so.",
    "Oh, don't mind me. I'll just figure it out myself.",
    "No problem. I'm used to it by now.",
    "Whatever makes you happy.",
    "That's fine. I didn't have plans anyway.",
    "I said I'm fine. Why do you keep asking?",
    "It doesn't matter. Forget I said anything.",
    "I'll just do it myself then. Like always.",
    "Don't worry about it. I'll handle it. Alone. Again.",
    # Silent treatment / withholding
    "I'm not upset. I just don't feel like talking right now.",
    "Nothing's wrong. I just have nothing to say.",
    "I'm fine. I just need some space. For a while.",
    "You wouldn't understand anyway, so there's no point.",
    "I already told you everything's fine.",
    # Backhanded compliance
    "If that's what you think is right, go ahead.",
    "Sure, I'll do it your way. Since my way is apparently always wrong.",
    "I'll just go along with whatever you decide.",
    "Okay, I'll stay quiet since my opinion doesn't matter.",
    "I'll do it. Even though no one asked if I wanted to.",
    # Must-be-nice trope
    "Must be nice to never have to worry about things like that.",
    "Must be nice having people just do things for you.",
    "It must be great to always get exactly what you want.",
    "Must be nice to be able to just forget about it.",
    # Martyrdom
    "Don't worry about me. I'll be fine. Eventually.",
    "I'll just stay late and finish it. Someone has to.",
    "Go have fun. I'll take care of everything here, as usual.",
    "I'm used to not being a priority.",
    "It's okay. I never expect anything anymore.",
    "I'll just do everything by myself. That seems to be how this works.",
    # Veiled threats
    "I'll just go then, if that's how you feel.",
    "Fine. Maybe I'll just disappear and see if anyone notices.",
    "I guess I'll just keep my thoughts to myself from now on.",
    "Maybe it's better if I just stop trying altogether.",
    # Pointed politeness
    "Oh, so NOW you care? Interesting.",
    "Oh, suddenly you have time for me.",
    "Wow, you remembered I exist.",
    "Oh, I didn't realize my feelings were up for debate.",
    "How thoughtful of you to finally check in.",
    # Indirect blame
    "If you had listened the first time, we wouldn't be in this situation.",
    "I'm not saying it's your fault. I'm just saying it happened after you left.",
    "I didn't say anything because I knew you'd react this way.",
    "It doesn't matter. You'll do what you want anyway.",
    "I just hope you're happy with the choice you made.",
    "Not that it matters what I think.",
    "I'm sure you already know what's best.",
    "Whatever. I hope it goes the way you expect.",
    "It's not my place to say anything, clearly.",
    "I'll just keep my concerns to myself since they're not welcome.",
    # More examples
    "Sure, go hang out with them. I'll just sit here.",
    "No, really, go ahead. I'll figure it out.",
    "It's fine. I only waited two hours.",
    "Don't worry, I'm sure they'll notice eventually.",
    "I'm glad someone's happy about this.",
    "Oh great, another thing I have to deal with.",
    "I'm fine. Why wouldn't I be fine?",
    "Not that you asked, but I'm doing terribly.",
    "I'm sure you're very busy. Too busy for this, apparently.",
    "Don't let me stop you.",
    "I guess my feelings just don't factor in here.",
    "I wasn't going to say anything but since you asked, whatever.",
    "Interesting that you'd bring that up now.",
    "I'll just add it to the list of things I handle alone.",
    "I hope you get what you're looking for. Truly.",
]

GASLIGHTING = [
    # Denying reality
    "I never said that. You're making things up.",
    "That's not what happened at all. You're remembering it wrong.",
    "I never did that. I don't know what you're talking about.",
    "You're imagining things again.",
    "That conversation never happened.",
    "I already told you that. You just weren't listening.",
    "I never promised that. You heard what you wanted to hear.",
    "That didn't happen the way you're describing it.",
    "You always twist things to make me look bad.",
    "You're rewriting history again.",
    # Questioning sanity / perception
    "You're being way too sensitive about this.",
    "You're overreacting. This is not a big deal.",
    "You're so dramatic. No one else thinks this is a problem.",
    "You're paranoid. No one is out to get you.",
    "You're too emotional to see this clearly right now.",
    "You always blow things out of proportion.",
    "You're acting crazy right now.",
    "You need to calm down and think rationally.",
    "You need to see a therapist if you think that's what happened.",
    "You're just being insecure. It's exhausting.",
    "That's not abuse. You're being delusional.",
    "You're so sensitive, you can't handle a normal conversation.",
    # Deflecting and invalidating
    "Everyone else thinks you're the problem here.",
    "I'm the only one who puts up with you.",
    "No one else would be as patient with you as I am.",
    "Ask anyone — they'll tell you you're wrong.",
    "Even your friends agree that you overreact.",
    "Everyone thinks you're too much to handle.",
    # Minimizing harm
    "I was just joking. Why can't you take a joke?",
    "Can't you take a joke? I was being sarcastic.",
    "You know I was just kidding, right? Stop being so sensitive.",
    "I barely even raised my voice. You're acting like it was abuse.",
    "I was just being honest. I didn't mean to hurt you.",
    "That wasn't even a fight. You're making a mountain out of a molehill.",
    "I said one thing. One. And now you're making it a whole thing.",
    # Shifting blame
    "If you hadn't made me so angry, I wouldn't have said that.",
    "You pushed me to say that. This is on you.",
    "I only acted that way because of how you were behaving.",
    "You brought this on yourself.",
    "You're the reason this relationship is so difficult.",
    "Everything was fine until you started acting like this.",
    "I wouldn't have to say these things if you just listened.",
    # Isolation tactics
    "Your friends are a bad influence on you.",
    "They don't actually care about you the way I do.",
    "Your family just doesn't understand our relationship.",
    "You only think that because of what your friends told you.",
    "The people you trust are the ones lying to you.",
    # Confusion and self-doubt induction
    "You've always had a bad memory. You know that.",
    "You do this every time. This is a pattern with you.",
    "You always find a way to ruin things.",
    "You're always the victim, aren't you.",
    "You're never satisfied, no matter what I do.",
    "I literally cannot say anything around you.",
    "There's no point talking to you when you're like this.",
    "You always accuse me of things I didn't do.",
    "You're the problem in every relationship you've had.",
    # More varied phrasing
    "That's a lie and you know it.",
    "You completely misunderstood what I said, as usual.",
    "You're twisting my words to start a fight.",
    "I don't know why I bother explaining anything to you.",
    "You're so caught up in your feelings you can't see the truth.",
    "I never said I'd be there. You assumed that on your own.",
    "Stop telling people things that aren't true about me.",
    "You're so focused on being the victim you can't see reality.",
    "I've never once disrespected you. You can't name one time.",
    "That's literally in your head.",
]

SINCERE_EXTRA = [
    "I really appreciate you taking the time to explain that to me.",
    "I hear what you're saying and I want to understand better.",
    "Thank you for being honest with me about this.",
    "I'm sorry I hurt you. That wasn't my intention at all.",
    "I was wrong about that. You were right.",
    "I want to make sure you feel heard and supported.",
    "Can we talk about this when we've both had some time to cool down?",
    "I genuinely didn't know that bothered you. I'll be more mindful.",
    "I'm proud of you for how you handled that.",
    "I'm here for you. What do you need right now?",
    "That must have been really hard. I'm glad you told me.",
    "You did a great job today. Seriously.",
    "I value your perspective on this.",
    "Let's find a solution that works for both of us.",
    "I'm sorry for the misunderstanding. Let me clarify.",
    "You've grown so much this year. I notice it.",
    "I'm committed to doing better.",
    "How are you feeling about all of this? I want to know.",
    "I'll always be honest with you, even when it's uncomfortable.",
    "You matter to me and so does how you feel.",
    "Take your time. I'm not going anywhere.",
    "I can see this is really affecting you.",
    "I'll listen as long as you need to talk.",
    "You don't have to apologize for having feelings.",
    "That was a fair point. I hadn't considered it that way.",
    "I'd love to hear more about how you see it.",
    "I understand why you reacted the way you did.",
    "I'm not going to dismiss what you just told me.",
    "Thank you for trusting me with this.",
    "Let's make sure we're both on the same page.",
]

SARCASM_EXTRA = [
    "Oh brilliant, another meeting that could have been an email.",
    "Wow, what a revolutionary idea. Nobody has ever thought of that before.",
    "Oh sure, because everything you touch turns to gold.",
    "Great, exactly what I wanted — more work with no extra pay.",
    "Oh fantastic, my favorite — being interrupted mid-sentence.",
    "What a shocker. I'm absolutely stunned that this happened. Again.",
    "Oh wow, you actually showed up on time. Mark the calendar.",
    "Sure, because your track record of great ideas really speaks for itself.",
    "Oh absolutely, I'm sure this time will be completely different.",
    "Wow, thanks for that incredibly helpful non-answer.",
    "Oh interesting, I had no idea you had that many opinions about my life.",
    "Yeah no, that's definitely a great plan. Can't see how anything could go wrong.",
    "Wow, such empathy. I really feel understood right now.",
    "Oh I'm sure they'll appreciate your very unique brand of feedback.",
    "What a surprise, another broken promise. Didn't see that coming.",
    "Oh wow, you noticed I've been upset for three days. Gold star.",
    "Brilliant. Truly your best work.",
    "Sure, I'll just drop everything I'm doing. Because my time means nothing.",
    "Oh great, more unsolicited advice. Just what I needed.",
    "Wow, what a well-thought-out response. Really hit the nail on the head.",
]


def get_curated_data():
    """Returns list of (text, label_id) tuples."""
    data = []
    data += [(t, 2) for t in PASSIVE_AGGRESSIVE]
    data += [(t, 3) for t in GASLIGHTING]
    data += [(t, 0) for t in SINCERE_EXTRA]
    data += [(t, 1) for t in SARCASM_EXTRA]
    return data


LABEL2ID = {"Sincere": 0, "Sarcastic": 1, "Passive-Aggressive": 2, "Gaslighting": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

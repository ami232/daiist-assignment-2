# Assignment 2 Report

- Name: Nicolas Loescher Montal
- Student ID: 20322
- Email: nloescher.ieu2022@student.ie.edu
- Group: PPLEDBA 5A

## Business problem, and the cost of being wrong

Finding a flat in Madrid is hard. You spend hours scrolling through listings,
and half the time the filters don't match what's actually in the flat. A
landlord forgets to tick a box and the place never shows up in your search. Or
a listing promises something, you go to see it, and it isn't there. When good
flats go fast, every listing you never see is one you might have wanted.

Portals like Idealista get thousands of photos a day, but they only know
what's in a flat if the landlord fills in the form properly. The photos
usually show more than the form does. My model looks at a room photo and finds
27 kinds of fixtures, from stoves and toilets to beds and air conditioning.
A portal could use that to make filters more accurate and to flag listings
that don't match their photos.

For most fixtures, a wrong call doesn't matter much. Nobody filters for a
faucet. So I priced one decision that does matter: does this photo show air
conditioning (`ac`)? For each photo the system either matches it (the listing
shows up in AC searches), rejects it, or sends it to a moderator to check.

AC felt like the obvious choice. Anyone who has spent a summer in Madrid knows
a flat without it can be miserable, and plenty of flats here don't have it, so
people really do filter for it. Landlords also forget to mention it all the
time.

If the model misses AC, the flat disappears from AC searches and nobody ever
finds out. The landlord, who's the one paying the portal, loses those
renters. I put that at about EUR 25, roughly half a day of extra vacancy on a
EUR 1,500 a month flat. A false match is less bad. You open the listing, see
there's no unit, and move on, so I put it at about EUR 5. A miss is about 5
times worse, so the system should lean towards showing the listing or asking a
person rather than hiding it.

## Dataset

I used the Roboflow dataset `so-d4hcz/all_finalize`, version 3 (6,277
images). Each image is a wide-angle room photo from a property listing, with
fixtures labelled in 27 classes, including AC. That's very close to what a
portal receives.

The catch is that the photos are from Hong Kong, where a lot of AC units are
window units. In Madrid you mostly see split units on the wall, or ducted
systems where all you can see is a grille in the ceiling.

## What the dataset's composition implies

The test set is small: 264 images, and only 16 of them have AC. The training
set also looks bigger than it is, since each photo was copied three times with
small changes. AC units are small too, usually around 1.5% of the photo.

With 16 examples, the 50% recall I get could easily be anywhere from about 28%
to 72%. Most other classes are thin as well. Some photos also appear in both
training and test, and the test set has AC in 6% of photos when I'd expect
about 2% on a real portal, so I calculate all costs at 2%.

So I can't promise the model will work this well on Madrid photos, and the
exact thresholds rest on very few examples.

## Finetuning and baselines

I finetuned `yolo11n`, starting from a model pretrained on COCO. I used
detection because the AC unit is such a small part of the photo, and the box
shows a moderator exactly where to look. The model is small enough to train
on a laptop and cheap to run every day.

I trained it for 20 epochs, which took about two and a half hours. Across all
classes it scored mAP50 0.654 on the test set.

| Model | Recall | Specificity | Accuracy |
|---|---|---|---|
| My finetuned model (cut 0.65) | 43.8% | 99.2% | 95.8% |
| COCO model, no finetuning | 0% | 100% | 93.9% |
| Always say "no AC" | 0% | 100% | 93.9% |

Accuracy doesn't help much here. Saying "no AC" every time already gets 93.9%,
because most photos don't have it. That's why I look at recall and cost
instead. The COCO model doesn't know what AC is, so mine is the only one that
finds any.

## Per-class evaluation and failure modes

| Class | AP50 |
|---|---|
| toilet | 0.975 |
| stove | 0.868 |
| cupboard | 0.612 |
| ac | 0.356 |

The full table is in `artifacts/results.json` and on the dashboard.

The model is best at big bathroom fixtures. Small things like faucets and AC
are harder, and some classes barely appear in the data at all. Most mistakes
are the model missing something or seeing something unlabelled. Some of those
aren't really mistakes: the three photos where the model was most confident
about unlabelled AC all clearly have a unit in them.

What worries me most is split units. Six of the 16 AC photos got almost no
score, including one with a large split unit near the ceiling. That's exactly
what a typical Madrid flat looks like.

## Cost matrix and operating point

| Value | Setting | My reasoning |
|---|---|---|
| `COST_FN` | EUR 25 | About half a day of vacancy |
| `COST_FP` | EUR 5 | Wasted time for the renter |
| `BASE_RATE` | 2% | My guess for how many listing photos show AC |
| `COST_REVIEW` | EUR 0.04 | A moderator at ~EUR 15 an hour, 10 seconds a photo |
| `PHOTOS_PER_DAY` | 15,000 | My rough guess for new Madrid listing photos |

If the model decides everything on its own, the cheapest threshold costs
about EUR 321 per 1,000 photos. Letting a person check the unclear ones works
better. Below 0.05 gets rejected and everything above goes to a moderator.
Nothing gets matched automatically, because even the model's most confident
calls include photos labelled as having no AC. That costs about EUR 234 per
1,000 photos, about 27% less, and catches about 56% of AC photos. Reviewing is
so cheap compared to missing a flat that it makes sense to send anything
likely to a person.

If a missed flat costs a lot more than I think, or AC is much more common, it
becomes cheaper to have someone check almost everything. Those are the two
numbers I'd want to check first.

## What a person still has to check

About 9% of photos go to review, around 1,400 a day. That's just under 4
hours of work, less than one moderator, so it's very doable.

| Design | EUR per day |
|---|---|
| Model + review (chosen) | 3,514 |
| Person checks everything | 3,555 |
| Model decides alone | 4,812 |
| Do nothing | 7,500 |

It costs about the same as having someone look at every photo, but it needs
less than one moderator instead of seven, and it gets cheaper as the model
improves. It's better to only ask a person when the model isn't sure.

## Limitations and next steps

1. The photos are from Hong Kong. I'd label a few hundred Idealista photos and
   finetune on them.
2. There are only 16 AC test photos. I'd label more to get at least 50.
3. Some labels are missing. I'd go through the AC labels and re-run the
   analysis.
4. Six of the 16 AC photos get almost no score. I'd decide per listing instead
   of per photo, since AC often appears in several photos of the same flat.
5. Training was short. I'd train longer on a GPU with a slightly bigger model.
6. Most of the costs are my own guesses. I'd want real numbers from a portal.

## Generative AI use disclosure

I used Claude a little for the Gradio dashboard. Throughout the project I also
asked it questions about possible next steps, which helped me figure out where
to take things. I checked the numbers against the pipeline output and can
explain each decision.